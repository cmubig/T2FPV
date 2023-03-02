import torch
import torch.nn as nn
import numpy as np
from .feature_extractor import build_feature_extractor
from .bitrap_np import BiTraPNP
import torch.nn.functional as F
from torch.autograd import Variable

from vrnntools.trajpred_models.modeling.mlp import MLP
from vrnntools.utils.common import dotdict
from vrnntools.utils.adj_matrix import simple_distsim_adjs
from vrnntools.trajpred_models.modeling.gat import GAT, SpGAT
from vrnntools.utils.adj_matrix import ego_dists, simple_adjs, simple_distsim_adjs

class SGNet_CVAE(nn.Module):
    def __init__(self, args, device):
        super(SGNet_CVAE, self).__init__()
        self.device = device
        self.cvae = BiTraPNP(args)
        self.input_dim = args.input_dim # Input dim
        self.hidden_size = args.hidden_size # GRU hidden size
        self.enc_steps = args.enc_steps # observation step
        self.dec_steps = args.dec_steps # prediction step
        self.dropout = args.dropout
        self.layer_norm = 'layer_norm' in args and args.layer_norm
        self.feature_extractor = build_feature_extractor(args)
        self.pred_dim = args.pred_dim
        self.K = args.K
        self.map = False
        self.pred_dim = 2

        self.no_abs = args.no_abs if 'no_abs' in args else False
        self.use_corr = args.use_corr
        self.criterion = nn.MSELoss()
        if self.use_corr:
            # Simple Input: 
            # - pred xy_abs
            # Simple Output:
            # - corrected xy_abs (RMSE w/ gt_abs)
            feat_enc_offset = dotdict(args.feat_enc_offset)
            self.f_offset = MLP(feat_enc_offset, self.device)
            self.corr_dim = feat_enc_offset.in_size
            # self.corr_idxs = torch.arange(0, self.corr_dim, device=self.device) if 'corr_idxs' not in feat_enc_offset \
            #                      else torch.tensor(feat_enc_offset.corr_idxs, device=self.device).to(int)

            if 'feat_enc_resnet' in args:
                feat_enc_resnet = dotdict(args.feat_enc_resnet)
                self.f_resnet = MLP(feat_enc_resnet, self.device)
            else:
                self.f_resnet = None

            corr_enc = dotdict(args.corr_enc)
            self.corr_enc = MLP(corr_enc, self.device)

            corr_rnn = dotdict(args.corr_rnn)
            self.corr_num_layers = corr_rnn.num_layers
            self.corr_rnn_dim = corr_rnn.hidden_size
            self.corr_rnn = nn.GRU(corr_rnn.in_size, self.corr_rnn_dim, self.corr_num_layers)

            self.corr_dec_rnn = nn.GRU(corr_rnn.in_size, self.corr_rnn_dim, self.corr_num_layers)
            corr_dec = dotdict(args.corr_dec)
            self.corr_dec = MLP(corr_dec, self.device)
            self.offset_idxs = torch.tensor(args.idxs, dtype=torch.long).to(self.device)
            self.selective = 'selective_corr' in args and args.selective_corr

        if 'graph' in args:
            graph = dotdict(args.graph)
            self.sigma = graph.sigma
            if graph.type == "gat":
                self.graph = GAT(self.hidden_size, graph.graph_hid, self.hidden_size, graph.alpha, graph.nheads)
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
            # if self.layer_norm:
            #     self.lg = nn.Sequential(nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
            #                             nn.LayerNorm(self.hidden_size))
            # else:
            #     self.lg = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
            self.lg = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        else:
            self.sigma = None
            self.graph = None
            self.lg = None
        self.train_start_time = 0 if 'start_time' not in args else args.start_time

        # TODO: layer norm throughout? use MLPs rather than single layers?
        if 'feat_enc_resnet_main' in args:
            feat_enc_resnet = dotdict(args.feat_enc_resnet)
            embed_size = args.hidden_size
            self.feat_enc_resnet = MLP(feat_enc_resnet, device=self.device)
            self.combine = MLP(dotdict({'in_size': feat_enc_resnet.out_size + embed_size,
                                        'hidden_size': [embed_size],
                                        'out_size': embed_size,
                                        'dropout': 0.0}), device=self.device)
            # TODO: layer_norm here?
            def combine_input(input_x, input_resnet):
                # 1 layer: 4 -> 96 -> ReLu
                feat_x = self.feature_extractor(input_x)
                # 2 layers: 2048 -> 512 -> 96
                feat_resnet = self.feat_enc_resnet(input_resnet)
                # 192 wide
                combined = torch.cat([feat_x, feat_resnet], dim=-1)
                #return self.combine(combined)
                return combined
                # combined = torch.cat([feat_x, input_resnet], dim=-1)
                # return self.feat_enc_resnet(combined)
            self.combine_input = combine_input
        else:
            self.feat_enc_resnet = None
            self.combine_input = None

        # the predict shift is in meter
        self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                    self.pred_dim))   
        if self.layer_norm:
            self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    1),
                                                    nn.LayerNorm(1),
                                                    nn.ReLU(inplace=True))
            self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    1),
                                                    nn.LayerNorm(1),
                                                    nn.ReLU(inplace=True))

            self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size//4),
                                                    nn.LayerNorm(self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
            self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size),
                                                    nn.LayerNorm(self.hidden_size),
                                                    nn.ReLU(inplace=True))
            self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + args.LATENT_DIM,
                                                    self.hidden_size),
                                                    nn.LayerNorm(self.hidden_size),
                                                    nn.ReLU(inplace=True))
            self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.LayerNorm(self.hidden_size),
                                                    nn.ReLU(inplace=True))

            self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size//4),
                                                    nn.LayerNorm(self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
            self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                        self.hidden_size),
                                                    nn.LayerNorm(self.hidden_size),
                                                    nn.ReLU(inplace=True))
            self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size//4),
                                                    nn.LayerNorm(self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
            self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size//4),
                                                    nn.LayerNorm(self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        else:
            self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    1),
                                                    nn.ReLU(inplace=True))
            self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    1),
                                                    nn.ReLU(inplace=True))

            self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
            self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size),
                                                        nn.ReLU(inplace=True))
            self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + args.LATENT_DIM,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
            self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))

            self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size//4),
                                                        nn.ReLU(inplace=True))
            self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                        self.hidden_size),
                                                        nn.ReLU(inplace=True))
            self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size//4),
                                                        nn.ReLU(inplace=True))
            self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                        self.hidden_size//4),
                                                        nn.ReLU(inplace=True))

        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        if self.feat_enc_resnet is not None:
            self.traj_enc_cell = nn.GRUCell(self.hidden_size*2 + self.hidden_size//4, self.hidden_size)
        else:
            self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
    
    def SGE(self, goal_hidden):
        # initial goal input with zero
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        # initial trajectory tensor
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            # next step input is generate by hidden
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            # regress goal traj for loss
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)
       
        K = dec_hidden.shape[1]
        # TODO: also add in GAT here? As in A-VRNN...
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        for dec_step in range(self.dec_steps):
            # incremental goal for each time step
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            batch_traj = self.regressor(dec_hidden)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:,dec_step,:,:] = batch_traj
        return dec_traj

    def encoder(self, raw_inputs, raw_targets, traj_input, flow_input=None, start_index = 0, hist_adj=None):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        total_probabilities = traj_input.new_zeros((traj_input.size(0), self.enc_steps, self.K))
        total_KLD = 0
        for enc_step in range(start_index, self.enc_steps):
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            # Refine here
            if hist_adj is not None and self.graph is not None:
                hist_adj_t = hist_adj[enc_step]
                h_g = self.graph(traj_enc_hidden, hist_adj_t)
                traj_enc_hidden = self.lg(torch.cat([traj_enc_hidden, h_g], -1))
            enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            all_goal_traj[:,enc_step,:,:] = goal_traj
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            if self.training:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K, raw_targets[:,enc_step,:,:])
            else:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K)
            total_probabilities[:,enc_step,:] = probability
            total_KLD += KLD
            cvae_dec_hidden= self.cvae_to_dec_hidden(cvae_hidden)
            if self.map:
                map_input = flow_input
                cvae_dec_hidden = (cvae_dec_hidden + map_input.unsqueeze(1))/2
            all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
        return all_goal_traj, all_cvae_dec_traj, total_KLD, total_probabilities
            
    def forward(self, inputs, map_mask=None, targets = None, start_index = 0, training=True,
                input_resnet=None, seq_start_end=None, hist_adj=None):
        # input given in T x B x d -> need to transpose
        inputs = inputs.permute(1, 0, 2)
        if input_resnet is not None:
            input_resnet = input_resnet.permute(1, 0, 2)
        self.training = training
        if torch.is_tensor(start_index):
            start_index = start_index[0].item()
        if self.feat_enc_resnet is not None:
            traj_input_temp = self.combine_input(inputs[:,start_index:,:], input_resnet[:,start_index:,:])
        else:
            # One layer: 4 -> 96 -> ReLu
            traj_input_temp = self.feature_extractor(inputs[:,start_index:,:])
        traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
        traj_input[:,start_index:,:] = traj_input_temp
        all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, None, start_index, 
                                                                                  hist_adj=hist_adj)
        return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities 

    def train_correction(self, hist_abs_gt, hist_yaw_gt, hist_abs_pred, hist_yaw_pred, hist_resnet, hist_seq_start_end, hist_valid):
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

            x_dec_feat_t = self.f_offset(x_dec_t)
            if self.selective:
                offset_pred_t = offset_pred[t]
                f_offset_pred_t = self.f_offset(offset_pred_t)
                x_dec_feat_t[hist_valid[t] == 1] = f_offset_pred_t[hist_valid[t] == 1]
                # Apply RMSE here...
                temp_mse = self._mse(x_dec_t[hist_valid[t] == 0], offset_gt_t[hist_valid[t] == 0])
                if not temp_mse.isnan():
                    MSE += temp_mse
                else:
                    MSE += torch.autograd.Variable(torch.zeros(1,).to(self.device), requires_grad = True)
            else:
                # Apply RMSE here...
                MSE += self._mse(x_dec_t, offset_gt_t)
            h_embedding = torch.cat([x_dec_feat_t], 1).unsqueeze(0)
            _, h = self.corr_dec_rnn(h_embedding, h)

            # compute losses
            if MSE.isnan():
                import pdb; pdb.set_trace()
        return MSE
    
    def infer_correction(self, hist_abs_pred, hist_yaw_pred, hist_resnet, hist_seq_start_end, hist_valid):
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
            if self.selective:
                offset_pred_t = offset_pred[t]
                f_offset_pred_t = self.f_offset(offset_pred_t)
                x_dec_feat_t[hist_valid[t] == 1] = f_offset_pred_t[hist_valid[t] == 1]
            h_embedding = torch.cat([x_dec_feat_t], 1).unsqueeze(0)
            _, h = self.corr_dec_rnn(h_embedding, h)

        # Need to change samples to absolute
        # TODO: use the relative only?
        if self.no_abs:
            samples = torch.cumsum(samples[..., -2:], dim=0) + xy_pred[0]
        return samples

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