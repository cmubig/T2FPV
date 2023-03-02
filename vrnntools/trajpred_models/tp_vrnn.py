# ------------------------------------------------------------------------------
# @file:    tp_vrnn.py
# @brief:   This class implements a simple CVAE-RNN-based trajectory prediction 
#           module.
#           Code based on: https://github.com/alexmonti19/dagnet
# @author:  Ingrid Navarro, Ben Stoler
# @date:    Last modified on August 3rd, 2022
# ------------------------------------------------------------------------------

import json
import logging
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from typing import Tuple

# local includes
from vrnntools.utils.common import dotdict
from vrnntools.utils.adj_matrix import ego_dists, simple_adjs, simple_distsim_adjs

from vrnntools.trajpred_models.modeling.mlp import MLP

logger = logging.getLogger(__name__)

seed = 1
np.random.seed(seed)

class TrajPredVRNN(nn.Module):
    """ A class that implements trajectory prediction model using a VRNN """
    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        """ Initializes the trajectory prediction network.
        Inputs:
        -------
        config[dict]: dictionary containing all configuration parameters.
        device[str]: device name used by the module. By default uses cuda:0. 
        """
        self._config = config
        super(TrajPredVRNN, self).__init__()
        logger.debug("{} configuration:\n{}".format(
            self.name, json.dumps(self.config, indent=2)))

        self.device = device
        logger.info(f"{self.name} uses torch.device({self.device})")
        
        self.dim = self.config.dim
        
        self.criterion = nn.MSELoss()
        
        # ----------------------------------------------------------------------
        # Model
        
        # x - feature extractor
        feat_enc = dotdict(self.config.feat_enc_x)
        self.f_x = MLP(feat_enc, self.device)
        self.f_x_out_size = feat_enc.out_size

        if 'feat_enc_resnet' in self.config:
            feat_enc_resnet = dotdict(self.config.feat_enc_resnet)
            self.f_resnet = MLP(feat_enc_resnet, self.device)
        else:
            self.f_resnet = None

        if 'feat_enc_ego_abs' in self.config:
            feat_enc_ego_abs = dotdict(self.config.feat_enc_ego_abs)
            self.use_ego_inference = False if not 'use_inference' in feat_enc_ego_abs \
                                           else feat_enc_ego_abs.use_inference
            self.ego_abs_time = 0 if 'time_offset' not in feat_enc_ego_abs \
                                  else feat_enc_ego_abs.time_offset
            self.f_ego_abs = MLP(feat_enc_ego_abs, self.device)
        else:
            self.f_ego_abs = None
            self.use_ego_inference = False
            self.ego_abs_time = 0
        
        self.train_start_time = 0 if 'start_time' not in self.config else self.config.start_time
        
        # x - encoder
        enc = dotdict(self.config.encoder)
        self.enc = MLP(enc, self.device)
        self.enc_out_size = enc.out_size
        assert self.enc_out_size % 2 == 0, \
            f"Encoder's output size must be divisible by 2"
        self.z_dim = int(self.enc_out_size / 2)

        # x - prior
        self.prior = MLP(dotdict(self.config.prior), self.device)

        # x - feature 
        self.f_z = MLP(dotdict(self.config.feat_enc_z), self.device)
        
        # x - decoder
        self.dec = MLP(dotdict(self.config.decoder), self.device)

        # recurrent network 
        rnn = dotdict(self.config.rnn)
        self.num_layers = rnn.num_layers
        self.rnn_dim = rnn.hidden_size
        self.rnn = nn.GRU(rnn.in_size, self.rnn_dim, self.num_layers)


        if 'graph' in self.config:
            graph = dotdict(self.config.graph)
            self.sigma = graph.sigma
            def avg_pool(features, adj):
                # Shape = B x B x d
                weighted = (features.unsqueeze(0).permute(2, 0, 1)*adj).permute(1, 2, 0)
                # Shape = B x d
                weighted = weighted.sum(dim=-2)/adj.sum(dim=-1).unsqueeze(-1)
                return weighted

            self.graph = avg_pool
            self.lg = nn.Linear(self.rnn_dim + self.rnn_dim, self.rnn_dim)
        else:
            self.graph = None
            self.sigma = 1.2
            self.lg = None

        logging.info(f"{self.name} architecture:\n{self}")
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def config(self) -> dict:
        return self._config
    

    def forward(
        self, hist: torch.tensor, hist_resnet: torch.tensor, hist_abs: torch.tensor,
        hist_seq_start_end: torch.tensor, hist_adj: torch.tensor, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor, Variable]:
        """ Forward propagation of observed trajectories.
        
        Inputs:
        -------
        hist[torch.tensor(hist_len, num_batch, dims)]: trajectory histories 
        kwargs: keyword-based arguments
            
        Outputs:
        --------
        KLD[torch.tensor]: accumulated KL divergence values
        NLL[torch.tensor]: accumulated Neg Log-Likelyhood values
        h[torch.tensor(num_rnn_layers, num_batch, r)]: torch.tensor
        """
        timesteps, num_agents, _ = hist.shape
      
        KLD = torch.zeros(1).to(self.device)
        NLL = torch.zeros(1).to(self.device)
        h = Variable(torch.zeros(
            self.num_layers, num_agents, self.rnn_dim)).to(self.device)
        
        # TODO: Add an ego_relative tensor, rather than just abs
        # Variables: 
        # - use at t or t+1
        # - encoder only, or encoder + prior + etc.
        # - incorporate w/ social pooling?
        # - incorporate w/ resnet (?)
        # - hidden state refinement?
        hist_ego_abs = ego_dists(hist_abs, hist_seq_start_end)

        for t in range(self.train_start_time, timesteps):
            # x - extract features at step t
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 

            if self.f_resnet is not None:
                resnet_t = hist_resnet[t]
                f_resnet_t = self.f_resnet(resnet_t)
            else:
                f_resnet_t = torch.empty((f_x_t.shape[0], 0), device=f_x_t.device)

            if self.f_ego_abs is not None:
                ego_abs_t = hist_ego_abs[t - self.ego_abs_time]
                f_ego_abs_t = self.f_ego_abs(ego_abs_t)
            else:
                f_ego_abs_t = torch.empty((f_x_t.shape[0], 0), device=f_x_t.device)

            # x - encode step t (encoder)
            x_enc_embedding = torch.cat([f_x_t, f_ego_abs_t, f_resnet_t, h[-1]], 1)
            x_enc_t = self.enc(x_enc_embedding)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            f_ego_abs_t = f_ego_abs_t if self.use_ego_inference \
                          else torch.empty((f_x_t.shape[0], 0), device=f_x_t.device)

            # x - encode step t (prior)
            prior_embedding = torch.cat([f_ego_abs_t, h[-1]], 1)
            x_prior_t = self.prior(prior_embedding)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # z - sample from latent space 
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # z - extract feature at step t
            f_z_t = self.f_z(z_t)

            # z - decode step t to generate x_t
            x_dec_embedding = torch.cat([f_z_t, f_ego_abs_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embedding)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # recurrence
            h_embedding = torch.cat([f_x_t, f_z_t, f_ego_abs_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

            # Hidden state refinement
            if self.graph is not None:
                adj_t = hist_adj[t]
                h_g = self.graph(h[-1], adj_t)
                h_refined = self.lg(torch.cat([h[-1], h_g], -1))
                h = torch.stack([h[0], h_refined])

            # compute losses
            KLD += self._kld(
                x_enc_mean_t, x_enc_logvar_t, x_prior_mean_t, x_prior_logvar_t)
            NLL += self._nll_gauss(x_dec_mean_t, x_dec_logvar_t, x_t)
            if KLD.isnan() or NLL.isnan():
                import pdb; pdb.set_trace()

        return KLD, NLL, h

    def inference(self, fut_len: int, h: Variable, hist_abs: torch.tensor, hist_seq_start_end: torch.tensor,
                  **kwargs) -> torch.tensor:
        """ Inference (sampling) trajectories.
        
        Inputs:
        -------
        fut_len[int]: length of the predicted trajectory
        h[torch.Variable(rnn_layers, num_batch, dim)]: torch.Variable 
        kwargs: any other keyword-based arguments
        
        Outputs:
        --------
        sample[torch.tensor(fut_len, num_batch, dims)]: predicted trajectories
        """
        _, num_agents, _ = h.shape

        samples = torch.zeros(fut_len, num_agents, self.dim).to(self.device)
        last_pos = hist_abs[-1]

        seq_adj = simple_adjs(last_pos.unsqueeze(0), hist_seq_start_end)[0]
        with torch.no_grad():
            for t in range(fut_len):
                
                hist_ego_abs = ego_dists(last_pos.unsqueeze(0), hist_seq_start_end)
                if self.f_ego_abs is not None and self.use_ego_inference:
                    ego_abs_t = hist_ego_abs[0]
                    f_ego_abs_t = self.f_ego_abs(ego_abs_t)
                else:
                    f_ego_abs_t = torch.empty((num_agents, 0), device=h.device)

                # x - encode hidden state to generate latent space (prior)
                prior_embedding = torch.cat([f_ego_abs_t, h[-1]], 1)
                x_prior_t = self.prior(prior_embedding)
                x_prior_mean_t = x_prior_t[:, :self.z_dim]
                x_prior_logvar_t = x_prior_t[:, self.z_dim:]

                # z - sample from latent space 
                z_t = self._reparameterize(x_prior_mean_t, x_prior_logvar_t)
                
                # z - extract feature at step t
                f_z_t = self.f_z(z_t)

                # z - decode step t to generate x_t
                x_dec_embedding = torch.cat([f_z_t, f_ego_abs_t, h[-1]], 1)
                x_dec_t = self.dec(x_dec_embedding)
                x_dec_mean_t = x_dec_t[:, :self.dim]
                
                # (N, D)
                samples[t] = x_dec_mean_t.data
                last_pos = samples[t] + last_pos

                # x - extract features from decoded latent space (~ 'x')
                f_x_t = self.f_x(x_dec_mean_t)

                # recurrence
                h_embedding = torch.cat([f_x_t, f_z_t, f_ego_abs_t], 1).unsqueeze(0)
                _, h = self.rnn(h_embedding, h)

                # Hidden state refinement
                if self.graph is not None:
                    adj_t = simple_distsim_adjs(last_pos.unsqueeze(0), hist_seq_start_end, self.sigma, seq_adj=seq_adj)[0]
                    h_g = self.graph(h[-1], adj_t)
                    h_refined = self.lg(torch.cat([h[-1], h_g], -1))
                    h = torch.stack([h[0], h_refined])

        return samples
    
    def _reparameterize(
        self, mean: torch.tensor, log_var: torch.tensor
    ) -> torch.tensor:
        """ Generates a sample z for the decoder using the mean, logvar parameters
        outputed by the encoder (during training) or prior (during inference). 
            z = mean + sigma * eps
        See: https://www.tensorflow.org/tutorials/generative/cvae
        
        Inputs:
        -------
        mean[torch.tensor]: mean of a Gaussian distribution 
        log_var[torch.tensor]: standard deviation of a Gaussian distribution.
                
        Outputs:
        --------
        z[torch.tensor]: sampled latent value. 
        """
        logvar = torch.exp(log_var * 0.5).to(self.device)
        # eps is a random noise
        eps = torch.rand_like(logvar).to(self.device)
        return eps.mul(logvar).add(mean)

    def _kld(
        self, mean_enc: torch.tensor, logvar_enc: torch.tensor, 
        mean_prior: torch.tensor, logvar_prior: torch.tensor
    ) -> torch.tensor:
        """ KL Divergence between the encoder and prior distributions:
            x1 = log(sigma_p / sigma_e)
            x2 = sigma_m ** 2 / sigma_p ** 2
            x3 = (mean_p - mean_e) ** 2 / sigma_p ** 2
            KL(p, q) = 0.5 * (x1 + x2 + x3 - 1)
        See: https://stats.stackexchange.com/questions/7440/ \
                kl-divergence-between-two-univariate-gaussians
        
        Inputs:
        -------
        mean_enc[torch.tensor]: encoder's mean at time t. 
        logvar_enc[torch.tensor]: encoder's variance at time t.
        mean_prior[torch.tensor]: prior's mean at time t. 
        logvar_prior[torch.tensor]: prior's variance at time t.
        
        Outputs:
        --------
        kld[torch.tensor]: Kullback-Leibler divergence between the prior and
        encoder's distributions time t. 
        """
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) /
                       (torch.exp(logvar_prior)), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def _nll_gauss(
        self, mean: torch.tensor, logvar: torch.tensor, x: torch.tensor
    ) -> torch.tensor:
        """ Negative Log-Likelihood with Gaussian.
            x1 = (x - mean) ** 2 / var
            x2 = logvar 
            x3 = const = 1 + log(2*pi)
            nll = 0.5 * (x1 + x2 + x3)
        See: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        
        Inputs:
        -------
        mean[torch.tensor]: decoder's mean at time t.
        logvar[torch.tensor]: decoder's variance a time t.
        x[torch.tensor]: ground truth X at time t.
        
        Outpus:
        -------
        nll[torch.tensor]: Gaussian Negative Log-Likelihood at time t. 
        """
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll

    def _mse(self, pred_x, gt_x) -> torch.tensor:
        """ Mean Squared Error between ground truth tensor (gt_x) and predicted
        tensor (pred_x). 
        Inputs:
        -------
        pred_x[torch.tensor]: predicted patterns
        gt_x[torch.tensor]: Ground truth patterns
        
        Outpus:
        --------
        mse[torch.Float]: mean squared error value
        """
        return torch.sqrt(self.criterion(pred_x, gt_x))
    