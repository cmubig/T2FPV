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
from vrnntools.trajpred_models.modeling.gat import GAT, SpGAT

logger = logging.getLogger(__name__)

seed = 1
np.random.seed(seed)

class TrajPredEgoAVRNN(nn.Module):
    """ A class that implements trajectory prediction model using a VRNN """
    def __init__(self, config: dict, device: str = "cuda:0") -> None:
        """ Initializes the trajectory prediction network.
        Inputs:
        -------
        config[dict]: dictionary containing all configuration parameters.
        device[str]: device name used by the module. By default uses cuda:0. 
        """
        self._config = config
        super(TrajPredEgoAVRNN, self).__init__()
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

        self.use_corr = self.config.use_corr
        self.no_abs = self.config.no_abs if 'no_abs' in self.config else False
        if self.use_corr:
            # Simple Input: 
            # - pred xy_abs
            # Simple Output:
            # - corrected xy_abs (RMSE w/ gt_abs)
            feat_enc_offset = dotdict(self.config.feat_enc_offset)
            self.f_offset = MLP(feat_enc_offset, self.device)
            self.corr_dim = feat_enc_offset.in_size
            # self.corr_idxs = torch.arange(0, self.corr_dim, device=self.device) if 'corr_idxs' not in feat_enc_offset \
            #                      else torch.tensor(feat_enc_offset.corr_idxs, device=self.device).to(int)

            if 'feat_enc_resnet' in self.config:
                feat_enc_resnet = dotdict(self.config.feat_enc_resnet)
                self.f_resnet = MLP(feat_enc_resnet, self.device)
            else:
                self.f_resnet = None

            corr_enc = dotdict(self.config.corr_enc)
            self.corr_enc = MLP(corr_enc, self.device)

            corr_rnn = dotdict(self.config.corr_rnn)
            self.corr_num_layers = corr_rnn.num_layers
            self.corr_rnn_dim = corr_rnn.hidden_size
            self.corr_rnn = nn.GRU(corr_rnn.in_size, self.corr_rnn_dim, self.corr_num_layers)

            self.corr_dec_rnn = nn.GRU(corr_rnn.in_size, self.corr_rnn_dim, self.corr_num_layers)
            corr_dec = dotdict(self.config.corr_dec)
            self.corr_dec = MLP(corr_dec, self.device)
            self.offset_idxs = torch.tensor(self.config.idxs, dtype=torch.long).to(self.device)

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

        # GAT and hidden state refinement
        graph = dotdict(self.config.graph)
        self.sigma = graph.sigma
        # Graphs take two args: features (B x d), adjacency (B x B)
        if graph.type == "sp_gat":
            self.graph = SpGAT(self.rnn_dim, graph.graph_hid, self.rnn_dim, graph.dropout, graph.alpha, graph.nheads)
        elif graph.type == "gat":
            self.graph = GAT(self.rnn_dim, graph.graph_hid, self.rnn_dim, graph.alpha, graph.nheads)
        elif graph.type == "pool":
            def avg_pool(features, adj):
                # Shape = B x B x d
                weighted = (features.unsqueeze(0).permute(2, 0, 1)*adj).permute(1, 2, 0)
                # Shape = B x d
                weighted = weighted.sum(dim=-2)/adj.sum(dim=-1).unsqueeze(-1)
                return weighted

            self.graph = avg_pool
        else:
            assert False, 'Invalid graph type provided'
        self.lg = nn.Linear(self.rnn_dim + self.rnn_dim, self.rnn_dim)

        logging.info(f"{self.name} architecture:\n{self}")
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def config(self) -> dict:
        return self._config

    def train_correction(self, hist_abs_gt, hist_yaw_gt, hist_abs_pred, hist_yaw_pred, hist_resnet, hist_seq_start_end):
        """ Trains the correction module for detections; requires no nan values
        """
        timesteps, num_agents, _ = hist_abs_gt.shape
        MSE = torch.zeros(1).to(self.device)
        h = Variable(torch.zeros(
            self.corr_num_layers, num_agents, self.corr_rnn_dim)).to(self.device)

        rel_gt = torch.zeros(hist_abs_gt.shape).to(hist_abs_gt.device)
        rel_gt[1:] = hist_abs_gt[1:] - hist_abs_gt[:-1]

        rel_pred = torch.zeros(hist_abs_pred.shape).to(hist_abs_pred.device)
        rel_pred[1:] = hist_abs_pred[1:] - hist_abs_pred[:-1]

        # https://stackoverflow.com/a/7869457
        # TODO: diff from beginning only?
        xy_gt = hist_abs_gt
        offset_xy_gt = ego_dists(hist_abs_gt, hist_seq_start_end)
        offset_yaw_gt = torch.deg2rad((180 + ego_dists(hist_yaw_gt, hist_seq_start_end)) % 360 - 180)
        offset_yaw_gt = torch.stack([torch.cos(offset_yaw_gt), torch.sin(offset_yaw_gt)], dim=-1)
        if self.no_abs:
            offset_gt = torch.cat([xy_gt - xy_gt[0], offset_xy_gt, offset_yaw_gt, rel_gt], dim=-1)
        else:
            offset_gt = torch.cat([xy_gt, offset_xy_gt, offset_yaw_gt, rel_gt], dim=-1)
        #offset_gt = offset_gt[..., :self.corr_dim]
        offset_gt = offset_gt[..., self.offset_idxs]

        xy_pred = hist_abs_pred
        offset_xy_pred = ego_dists(hist_abs_pred, hist_seq_start_end)
        offset_yaw_pred = torch.deg2rad((180 + ego_dists(hist_yaw_pred, hist_seq_start_end)) % 360 - 180)
        offset_yaw_pred = torch.stack([torch.cos(offset_yaw_pred), torch.sin(offset_yaw_pred)], dim=-1)
        if self.no_abs:
            offset_pred = torch.cat([xy_pred - xy_pred[0], offset_xy_pred, offset_yaw_pred, rel_pred], dim=-1)
        else:
            offset_pred = torch.cat([xy_pred, offset_xy_pred, offset_yaw_pred, rel_pred], dim=-1)
        #offset_pred = offset_pred[..., :self.corr_dim]
        offset_pred = offset_pred[..., self.offset_idxs]

        # For now, just do ego resnet view...
        #ego_resnet = torch.cat([torch.stack([hist_resnet[:, start, :] for _ in range(end - start)], dim=1) for start, end in hist_seq_start_end], dim=1)
        ego_resnet = hist_resnet
        ego_idxs = hist_seq_start_end[:, 0]
        det_mask = torch.ones((num_agents,), dtype=torch.bool, device=self.device)
        det_mask[ego_idxs] = False
        for t in range(self.train_start_time, timesteps):
            offset_pred_t = offset_pred[t]
            f_offset_pred_t = self.f_offset(offset_pred_t)
            if self.f_resnet is not None:
                f_resnet_t = self.f_resnet(ego_resnet[t])
            else:
                f_resnet_t = torch.empty((offset_pred_t.shape[0], 0), device=h.device)

            # Simple embedding:
            x_enc_embedding = torch.cat([f_offset_pred_t, f_resnet_t, h[-1]], 1)
            x_corr_t = self.corr_enc(x_enc_embedding)

            h_embedding = torch.cat([x_corr_t], 1).unsqueeze(0)
            _, h = self.corr_rnn(h_embedding, h)

        for t in range(self.train_start_time, timesteps):
            offset_gt_t = offset_gt[t]
            x_dec_embedding = torch.cat([h[-1]], 1)
            x_dec_t = self.corr_dec(x_dec_embedding)

            # Apply RMSE here...
            MSE += self._mse(x_dec_t, offset_gt_t)
            x_dec_feat_t = self.f_offset(x_dec_t)
            h_embedding = torch.cat([x_dec_feat_t], 1).unsqueeze(0)
            _, h = self.corr_dec_rnn(h_embedding, h)

            # compute losses
            if MSE.isnan():
                import pdb; pdb.set_trace()
        return MSE
    
    def infer_correction(self, hist_abs_pred, hist_yaw_pred, hist_resnet, hist_seq_start_end):
        timesteps, num_agents, _ = hist_abs_pred.shape
        h = Variable(torch.zeros(
            self.corr_num_layers, num_agents, self.corr_rnn_dim)).to(self.device)

        rel_pred = torch.zeros(hist_abs_pred.shape).to(hist_abs_pred.device)
        rel_pred[1:] = hist_abs_pred[1:] - hist_abs_pred[:-1]

        # https://stackoverflow.com/a/7869457
        xy_pred = hist_abs_pred
        offset_xy_pred = ego_dists(hist_abs_pred, hist_seq_start_end)
        offset_yaw_pred = torch.deg2rad((180 + ego_dists(hist_yaw_pred, hist_seq_start_end)) % 360 - 180)
        offset_yaw_pred = torch.stack([torch.cos(offset_yaw_pred), torch.sin(offset_yaw_pred)], dim=-1)
        if self.no_abs:
            offset_pred = torch.cat([xy_pred - xy_pred[0], offset_xy_pred, offset_yaw_pred, rel_pred], dim=-1)
        else:
            offset_pred = torch.cat([xy_pred, offset_xy_pred, offset_yaw_pred, rel_pred], dim=-1)
        #offset_pred = offset_pred[..., :self.corr_dim]
        offset_pred = offset_pred[..., self.offset_idxs]

        # For now, just do ego resnet view...
        #ego_resnet = torch.cat([torch.stack([hist_resnet[:, start, :] for _ in range(end - start)], dim=1) for start, end in hist_seq_start_end], dim=1)
        ego_resnet = hist_resnet
        samples = torch.zeros(timesteps, num_agents, offset_pred.shape[-1]).to(self.device)
        for t in range(self.train_start_time, timesteps):
            offset_pred_t = offset_pred[t]
            f_offset_pred_t = self.f_offset(offset_pred_t)
            if self.f_resnet is not None:
                f_resnet_t = self.f_resnet(ego_resnet[t])
            else:
                f_resnet_t = torch.empty((offset_pred_t.shape[0], 0), device=h.device)

            # Simple embedding:
            x_enc_embedding = torch.cat([f_offset_pred_t, f_resnet_t, h[-1]], 1)
            x_corr_t = self.corr_enc(x_enc_embedding)

            h_embedding = torch.cat([x_corr_t], 1).unsqueeze(0)
            _, h = self.corr_rnn(h_embedding, h)

        for t in range(self.train_start_time, timesteps):
            x_dec_embedding = torch.cat([h[-1]], 1)
            x_dec_t = self.corr_dec(x_dec_embedding)

            samples[t] = x_dec_t.data
            x_dec_feat_t = self.f_offset(x_dec_t)
            h_embedding = torch.cat([x_dec_feat_t], 1).unsqueeze(0)
            _, h = self.corr_dec_rnn(h_embedding, h)

        # Need to change samples to absolute
        # TODO: use the relative only?
        if self.no_abs:
            samples = torch.cumsum(samples[..., -2:], dim=0) + xy_pred[0]
        return samples

    

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

        for t in range(self.train_start_time, timesteps):
            # x - extract features at step t
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 

            # x - encode step t (encoder)
            x_enc_embedding = torch.cat([f_x_t, h[-1]], 1)
            x_enc_t = self.enc(x_enc_embedding)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            # x - encode step t (prior)
            prior_embedding = torch.cat([h[-1]], 1)
            x_prior_t = self.prior(prior_embedding)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]


            # z - sample from latent space 
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # z - extract feature at step t
            f_z_t = self.f_z(z_t)

            # z - decode step t to generate x_t
            x_dec_embedding = torch.cat([f_z_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embedding)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # recurrence
            h_embedding = torch.cat([f_x_t, f_z_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

            # graph refinement
            h_g = self.graph(h[-1], hist_adj[t])
            h_refined = self.lg(torch.cat([h[-1], h_g], -1))
            h = torch.stack([h[0], h_refined])

            # compute losses
            KLD += self._kld(x_enc_mean_t, x_enc_logvar_t, x_prior_mean_t, x_prior_logvar_t)
            NLL += self._nll_gauss(x_dec_mean_t, x_dec_logvar_t, x_t)
            if KLD.isnan() or NLL.isnan():
                import pdb; pdb.set_trace()

        return KLD, NLL, h

    def inference(self, fut_len: int, h: Variable, hist_abs: torch.tensor, hist_seq_start_end: torch.tensor,
                  num_samples: int,
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

        original_num_agents = h.shape[1]
        h = h.repeat(1, num_samples, 1)
        hist_abs = hist_abs.repeat(1, num_samples, 1)
        seq_start_end_add = torch.repeat_interleave(torch.arange(0, num_samples)*original_num_agents, hist_seq_start_end.shape[0], dim=0).to(hist_seq_start_end.device)
        hist_seq_start_end = hist_seq_start_end.repeat(num_samples, 1)
        hist_seq_start_end = hist_seq_start_end + seq_start_end_add.broadcast_to(2, seq_start_end_add.shape[-1]).permute(1, 0)

        
        seq_adj = simple_adjs(hist_abs[0].unsqueeze(0), hist_seq_start_end)[0]

        _, num_agents, _ = h.shape
        samples = torch.zeros(fut_len, num_agents, self.dim).to(self.device)
        last_pos = hist_abs[-1]

        # seq_adj = simple_adjs(last_pos.unsqueeze(0), hist_seq_start_end)[0]
        # ego_idxs = torch.tensor([p[0] for p in hist_seq_start_end], device=hist_abs.device)
        # det_idxs = torch.cat([torch.arange(p[0]+1, p[1]) for p in hist_seq_start_end])
        with torch.no_grad():
            for t in range(fut_len):
                # x - encode hidden state to generate latent space (prior)
                prior_embedding = torch.cat([h[-1]], 1)
                x_prior_t = self.prior(prior_embedding)
                x_prior_mean_t = x_prior_t[:, :self.z_dim]
                x_prior_logvar_t = x_prior_t[:, self.z_dim:]

                # z - sample from latent space 
                z_t = self._reparameterize(x_prior_mean_t, x_prior_logvar_t)
                
                # z - extract feature at step t
                f_z_t = self.f_z(z_t)

                # z - decode step t to generate x_t
                x_dec_embedding = torch.cat([f_z_t, h[-1]], 1)
                x_dec_t = self.dec(x_dec_embedding)
                x_dec_mean_t = x_dec_t[:, :self.dim]

                # (N, D)
                samples[t] = x_dec_mean_t.data
                last_pos = samples[t] + last_pos

                # x - extract features from decoded latent space (~ 'x')
                f_x_t = self.f_x(x_dec_mean_t)

                adj_t = simple_distsim_adjs(last_pos.unsqueeze(0), hist_seq_start_end, self.sigma, seq_adj=seq_adj)[0]

                # recurrence
                h_embedding = torch.cat([f_x_t, f_z_t], 1).unsqueeze(0)
                _, h = self.rnn(h_embedding, h)

                h_g = self.graph(h[-1], adj_t)
                h_refined = self.lg(torch.cat([h[-1], h_g], -1))
                h = torch.stack([h[0], h_refined])

        ret_samples = torch.zeros(num_samples, fut_len, original_num_agents, self.dim).to(self.device)
        for i in range(num_samples):
            offset = i*original_num_agents
            ret_samples[i] = samples[:, offset:offset+original_num_agents]
        return ret_samples
    
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
    