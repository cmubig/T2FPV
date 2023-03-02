# ------------------------------------------------------------------------------
# @file:    vrnn.py
# @brief:   This file contains the implementation of the VRNNTrainer class which
#           is used for training a Variational Recurrent Neural Network for 
#           trajectory prediction. 
# @author:  Ingrid Navarro, Ben Stoler
# @date:    Last modified on August 25th, 2022
# ------------------------------------------------------------------------------
import time
import logging
import os
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import vrnntools.utils.common as mutils

from vrnntools.trajpred_models.sgnet.tp_sgnet_cvae import SGNet_CVAE
from vrnntools.trajpred_trainers.base_trainer import BaseTrainer
from vrnntools.utils import metrics, processing
from vrnntools.utils.metrics import LossContainer, MetricContainer
from vrnntools.utils.metrics import thres_rmse, thres_cvae

from vrnntools.utils.adj_matrix import simple_distsim_adjs, simple_adjs

logger = logging.getLogger(__name__)

class SGNetCVAETrainer(BaseTrainer):
    """ A trainer class that implements trainer methods for the VRNN trajectory 
    prediction model. It inherits base methods for training and evaluating from 
    the BaseTrainer (See sprnn/trajpred_trainers/base_trainer.py). """
    def __init__(self, config: dict) -> None:
        """ Initializes the trainer.
        Inputs:
        -------
        config[dict]: a dictionary containing all configuration parameters.
        """
        super().__init__(config)
        self.setup()
        
        logger.info(f"{self.name} is ready!")
        
    def train_epoch(self, epoch: int, train_corr=False) -> dict:
        """ Trains one epoch.
        Inputs:
        -------
        epoch[int]: current training epoch
             
        Outputs:
        --------
        loss[dict]: a dictionary with all loss values computed during the epoch.
        """
        epoch_str = f"[{epoch}/{self.num_epoch}] Train"
        logger.info(f"Training epoch: {epoch_str}")
        self.model.train()
        
        batch_count = 0 
        batch_loss = 0

        self.train_losses.reset()
        assert not self.retrack, 'Retrack in test only'

        for i, batch in tqdm(enumerate(self.train_data), dynamic_ncols=True, desc=epoch_str, total=self.num_iter):
            if i >= self.num_iter:
                break
            
            self.optimizer.zero_grad()

            batch = [tensor.to(self.device) for tensor in batch]   
            hist_all, hist_resnet, hist_seq_start_end, gt_all, gt_seq_start_end = batch
            hist_abs, hist_yaw, hist_abs_img, hist_valid, hist_orig_id = \
                hist_all[..., :2], hist_all[..., 2], hist_all[..., 3:5], hist_all[..., 5], hist_all[..., 6]
            gt_abs, gt_yaw, gt_abs_img, gt_valid, gt_orig_id = \
                gt_all[..., :2], gt_all[..., 2], gt_all[..., 3:5], gt_all[..., 5], gt_all[..., 6]
            hist_len = hist_abs.shape[0]

            assert self.train_data.dataset.alignment, 'Unaligned data not supported'
            assert gt_valid.all(), 'Ground truth must be fully valid'
            assert not hist_all.isnan().any() and not hist_resnet.isnan().any(), 'NaN found in hist; interpolation needed?'
            assert (hist_seq_start_end == gt_seq_start_end).all(), 'Alignment must actually work...'
            # "Valid" means non-interpolated and non-missing

            if self.naomi:
                from helpers import infer_batch
                # TODO: use grad here or no?
                with torch.no_grad():
                    naomi_loss, naomi_ade, naomi_sample = infer_batch(self.naomi_model, hist_valid, hist_abs, gt_abs)
                hist_abs = naomi_sample

            if self.pretrained:
                if self.pretrained_module is not None or ((self.naomi) and self.pretrained_pred is not None):
                    mse_corr = torch.autograd.Variable(torch.zeros((1,), device=hist_abs.device), requires_grad=True)
                else:
                    mse_corr = self.model.train_correction(gt_abs[:hist_len], gt_yaw[:hist_len], hist_abs, hist_yaw, hist_resnet[:hist_len], hist_seq_start_end, hist_valid)
                goal = mse_corr
                dec = mse_corr
                kld = mse_corr
            elif self.model.use_corr:
                mse_corr = self.model.train_correction(gt_abs[:hist_len], gt_yaw[:hist_len], hist_abs, hist_yaw, hist_resnet[:hist_len], hist_seq_start_end, hist_valid)
            else:
                mse_corr = torch.zeros((1,), device=hist_abs.device)

            if self.pretrained_pred is None and self.pretrained:
                with torch.no_grad():
                    if self.pretrained_module is not None:
                        sample = self.pretrained_module.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end)
                    elif (self.naomi) and self.pretrained_pred is not None:
                        sample = hist_abs
                    else:
                        sample = self.model.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end, hist_valid)
                # Don't care about the yaw differences
                new_hist_abs = sample[:, :, :2]
                # for start, _ in hist_seq_start_end:
                #     new_hist_abs[:, start] = hist_abs[:, start]
                for i in range(hist_abs.shape[1]):
                    if 'repl_all' in self.config.TRAJECTORY and self.config.TRAJECTORY.repl_all:
                        if (hist_valid[:, i] == 0).any():
                            continue
                    new_hist_abs[hist_valid[:, i] == 1, i] = hist_abs[hist_valid[:, i] == 1, i]
                # TODO: e2e or no
                hist_abs = new_hist_abs

            # For now, try after potential corrections
            if self.smoothing:
                hist_abs = processing.holt_winters_smoothing(hist_abs)

            hist_rel = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_rel[1:] = hist_abs[1:] - hist_abs[:-1]
            num_batch = hist_rel.shape[1]

            if self.pretrained_pred is None:
                gt_rel = torch.zeros(gt_abs.shape).to(hist_abs.device)
                gt_rel[1:] = gt_abs[1:] - gt_abs[:-1]

                hist_acc = torch.zeros(hist_abs.shape).to(hist_abs.device)
                hist_acc[1:] = hist_rel[1:] - hist_rel[:-1]
                gt_acc = torch.zeros(gt_abs.shape).to(gt_abs.device)
                gt_acc[1:] = gt_rel[1:] - gt_rel[:-1]

                # 4d features
                hist_in = torch.cat([hist_abs, hist_rel, hist_acc], dim=-1)[..., -self.model.input_dim:]
                gt_in = torch.cat([gt_abs, gt_rel, gt_acc], dim=-1)[..., -self.model.input_dim:]
                gt_in = gt_in[:hist_len]

                target_traj = torch.zeros((gt_abs.shape[1], hist_len, self.fut_len, self.dim), device=hist_abs.device)
                combined_abs  = torch.cat([hist_abs, gt_abs[hist_len:]], dim=0)
                # Relative to last obs idx, cumulatively
                for obs_idx in range(hist_len):
                    target_traj[:, obs_idx] = (combined_abs[obs_idx+1:obs_idx+1+self.fut_len] - combined_abs[obs_idx]).permute(1, 0, 2)

                # Worry about nan's in the rmse function...
                # Assume aligned only in training, since targets cannot have nan...
                model_in = gt_in if self.train_gt else hist_in
                all_goal_traj, all_dec_traj, KLD_loss, _ = \
                    self.model(model_in, map_mask=None, targets=target_traj, start_index=0, training=True,
                            input_resnet=hist_resnet, seq_start_end=hist_seq_start_end)

                # Shapes: B x obs_len x pred_len x 2
                goal_ades = thres_rmse(all_goal_traj, target_traj, hist_seq_start_end, gt_seq_start_end, assume_aligned=True)
                dec_ades = thres_cvae(all_dec_traj, target_traj, hist_seq_start_end, gt_seq_start_end, assume_aligned=True)

                # ADE loss probably...
                loss = self.compute_loss(epoch=epoch, goal=goal_ades.sum()/self.fut_len, dec=dec_ades.sum()/self.fut_len, \
                                        kld=KLD_loss, mse=mse_corr)
            else:
                gt_rel = torch.zeros(gt_abs.shape).to(hist_abs.device)
                gt_rel[1:] = gt_abs[1:] - gt_abs[:-1]

                hist_acc = torch.zeros(hist_abs.shape).to(hist_abs.device)
                hist_acc[1:] = hist_rel[1:] - hist_rel[:-1]
                gt_acc = torch.zeros(gt_abs.shape).to(gt_abs.device)
                gt_acc[1:] = gt_rel[1:] - gt_rel[:-1]

                # 4d features
                hist_in = torch.cat([hist_abs, hist_rel, hist_acc], dim=-1)[..., -self.model.input_dim:]
                gt_in = torch.cat([gt_abs, gt_rel, gt_acc], dim=-1)[..., -self.model.input_dim:]
                gt_in = gt_in[:hist_len]

                target_traj = torch.zeros((gt_abs.shape[1], hist_len, self.fut_len, self.dim), device=hist_abs.device)
                combined_abs  = torch.cat([hist_abs, gt_abs[hist_len:]], dim=0)
                # Relative to last obs idx, cumulatively
                for obs_idx in range(hist_len):
                    target_traj[:, obs_idx] = (combined_abs[obs_idx+1:obs_idx+1+self.fut_len] - combined_abs[obs_idx]).permute(1, 0, 2)

                # Worry about nan's in the rmse function...
                # Assume aligned only in training, since targets cannot have nan...
                model_in = gt_in if self.train_gt else hist_in
                with torch.no_grad():
                    all_goal_traj, all_dec_traj, KLD_loss, _ = \
                        self.model(model_in, map_mask=None, targets=target_traj, start_index=0, training=True,
                                input_resnet=hist_resnet, seq_start_end=hist_seq_start_end)

                # Shapes: B x obs_len x pred_len x 2
                goal_ades = thres_rmse(all_goal_traj, target_traj, hist_seq_start_end, gt_seq_start_end, assume_aligned=True)
                dec_ades = thres_cvae(all_dec_traj, target_traj, hist_seq_start_end, gt_seq_start_end, assume_aligned=True)

                # ADE loss probably...
                loss = self.compute_loss(epoch=epoch, goal=goal_ades.sum()/self.fut_len, dec=dec_ades.sum()/self.fut_len, \
                                        kld=KLD_loss, mse=mse_corr)
            batch_loss += loss['Loss']
            batch_count += 1
    
            if batch_count >= self.batch_size:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                batch_loss = 0.0
                batch_count = 0
                
            self.train_losses.update([loss], num_batch)
        return self.train_losses.get_losses()

    def eval_epoch(self, epoch: int, train_corr=False, **kwargs) -> dict:
        """ Evaluates one epoch.
        Inputs:
        -------
        epoch[int]: current eval epoch.
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: a dictionary with all of losses computed during training. 
        """  
        self.model.eval()
        num_samples = kwargs.get('num_samples') if kwargs.get('num_samples') else 1
        
        self.eval_metrics.reset()
        assert not self.retrack, 'Retrack in test only'
        
        epoch_str = f"[{epoch}/{self.num_epoch}] Val"
        for i, batch in tqdm(enumerate(self.val_data), dynamic_ncols=True, desc=epoch_str, total=self.eval_num_iter):
            if i >= self.eval_num_iter:
                break

            batch = [tensor.to(self.device) for tensor in batch]   
            hist_all, hist_resnet, hist_seq_start_end, gt_all, gt_seq_start_end = batch
            hist_abs, hist_yaw, hist_abs_img, hist_valid, hist_orig_id = \
                hist_all[..., :2], hist_all[..., 2], hist_all[..., 3:5], hist_all[..., 5], hist_all[..., 6]
            gt_abs, gt_yaw, gt_abs_img, gt_valid, gt_orig_id = \
                gt_all[..., :2], gt_all[..., 2], gt_all[..., 3:5], gt_all[..., 5], gt_all[..., 6]
            hist_len = hist_abs.shape[0]

            assert self.val_data.dataset.alignment, 'Unaligned data not supported'
            assert gt_valid.all(), 'Ground truth must be fully valid'
            assert not hist_all.isnan().any() and not hist_resnet.isnan().any(), 'NaN found in hist; interpolation needed?'
            assert (hist_seq_start_end == gt_seq_start_end).all(), 'Alignment must actually work...'

            if self.naomi:
                from helpers import infer_batch
                # TODO: use grad here or no?
                with torch.no_grad():
                    naomi_loss, naomi_ade, naomi_sample = infer_batch(self.naomi_model, hist_valid, hist_abs, gt_abs)
                hist_abs = naomi_sample

            if self.pretrained:
                with torch.no_grad():
                    if self.pretrained_module is not None:
                        sample = self.pretrained_module.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end)
                    elif (self.naomi) and self.pretrained_pred is not None:
                        sample = hist_abs
                    else:
                        sample = self.model.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end, hist_valid)
                # Don't care about the yaw differences
                new_hist_abs = sample[:, :, :2]
                # for start, _ in hist_seq_start_end:
                #     new_hist_abs[:, start] = hist_abs[:, start]
                for i in range(hist_abs.shape[1]):
                    if 'repl_all' in self.config.TRAJECTORY and self.config.TRAJECTORY.repl_all:
                        if (hist_valid[:, i] == 0).any():
                            continue
                    new_hist_abs[hist_valid[:, i] == 1, i] = hist_abs[hist_valid[:, i] == 1, i]
                hist_abs = new_hist_abs
            elif self.model.use_corr:
                with torch.no_grad():
                    sample = self.model.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end, hist_valid)
                # Don't care about the yaw differences
                new_hist_abs = sample[:, :, :2]
                for start, _ in hist_seq_start_end:
                    new_hist_abs[:, start] = hist_abs[:, start]
                hist_abs = new_hist_abs
           
            # For now, try after potential corrections
            if self.smoothing:
                hist_abs = processing.holt_winters_smoothing(hist_abs)

            hist_rel = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_rel[1:] = hist_abs[1:] - hist_abs[:-1]
            num_batch = hist_rel.shape[1]

            hist_acc = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_acc[1:] = hist_rel[1:] - hist_rel[:-1]

            # 4d features
            hist_in = torch.cat([hist_abs, hist_rel, hist_acc], dim=-1)[..., -self.model.input_dim:]

            target_traj = torch.zeros((gt_abs.shape[1], hist_len, self.fut_len, self.dim), device=hist_abs.device)
            # Relative to last obs idx, cumulatively
            for obs_idx in range(hist_len):
                target_traj[:, obs_idx] = (gt_abs[obs_idx+1:obs_idx+1+self.fut_len] - gt_abs[obs_idx]).permute(1, 0, 2)
            
            # run forward propagation for the trajectory's history, assuming model is deterministic for warmup on obs
            with torch.no_grad():
                if self.pretrained_pred is not None:
                    all_goal_traj, all_dec_traj, KLD_loss, _ = \
                        self.pretrained_pred(hist_in, map_mask=None, targets=target_traj, start_index=0, training=False,
                                input_resnet=hist_resnet, seq_start_end=hist_seq_start_end)
                else:
                    all_goal_traj, all_dec_traj, KLD_loss, _ = \
                        self.model(hist_in, map_mask=None, targets=target_traj, start_index=0, training=False,
                                input_resnet=hist_resnet, seq_start_end=hist_seq_start_end)
                # Shapes: B x obs_len x pred_len x 2
                # convert the prediction to absolute coords
                # Shape = num_samples x N x B x d
                preds = hist_abs[-1] + all_dec_traj[:, -1].permute(2, 1, 0, 3)
                
            # compute best of num_samples
            self.eval_metrics.update(gt_abs[hist_len:], preds, hist_seq_start_end)

            assert not self.visualize, 'Visualize not yet supported'
            # TODO: save outputs (?) Maybe in test only
            if self.visualize and i % self.plot_freq == 0:
                # self.generate_outputs(
                #     hist_abs, fut_abs, preds, best_sample_idx, seq_start_end, 
                #     f"epoch-{epoch+1}_val-{i}", epoch)
                pass
                        
        metrics = self.eval_metrics.get_metrics()
        self_name = self.out.base.split('/')[-1].split('_rel_2d')[0]
        print(f'{self_name} {time.strftime("%x %X")}: ADE={metrics["MinADE"]:.3f}, FDE={metrics["MinFDE"]:.3f}'\
                f', ADEMed={metrics["MinADEMed"]:.3f}, FDEMed={metrics["MinFDEMed"]:.3f}')
        return metrics
    
    def test_epoch(self, epoch: int, train_corr=False, **kwargs) -> dict:
        """ Tests one epoch.
        Inputs:
        -------
        epoch[int]: current eval epoch.
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: a dictionary with all of losses computed during training. 
        """  
        self.model.eval()
        num_samples = kwargs.get('num_samples') if kwargs.get('num_samples') else 1
        
        self.eval_metrics.reset()
        
        epoch_str = f"[{epoch}/{self.num_epoch}] Test"
        for batch_idx, batch in tqdm(enumerate(self.test_data), dynamic_ncols=True, desc=epoch_str, total=self.test_num_iter):
            if batch_idx >= self.test_num_iter:
                break
            
            batch = [tensor.to(self.device) for tensor in batch]   
            hist_all, hist_resnet, hist_seq_start_end, gt_all, gt_seq_start_end = batch
            hist_abs, hist_yaw, hist_abs_img, hist_valid, hist_orig_id = \
                hist_all[..., :2], hist_all[..., 2], hist_all[..., 3:5], hist_all[..., 5], hist_all[..., 6]
            gt_abs, gt_yaw, gt_abs_img, gt_valid, gt_orig_id = \
                gt_all[..., :2], gt_all[..., 2], gt_all[..., 3:5], gt_all[..., 5], gt_all[..., 6]
            hist_len = hist_abs.shape[0]

            assert self.test_data.dataset.alignment, 'Unaligned data not supported'
            assert gt_valid.all(), 'Ground truth must be fully valid'
            assert not hist_all.isnan().any() and not hist_resnet.isnan().any(), 'NaN found in hist; interpolation needed?'
            assert (hist_seq_start_end == gt_seq_start_end).all(), 'Alignment must actually work...'

            tensors_out = {}
            tensors_out[f'seq_start_end'] = hist_seq_start_end
            tensors_out[f'hist_valid'] = hist_valid
            tensors_out[f'hist_abs'] = hist_abs
            tensors_out[f'hist_yaw'] = hist_yaw
            tensors_out[f'gt_abs'] = gt_abs
            tensors_out[f'gt_yaw'] = gt_yaw

            if self.naomi:
                from helpers import infer_batch
                # TODO: use grad here or no?
                with torch.no_grad():
                    naomi_loss, naomi_ade, naomi_sample = infer_batch(self.naomi_model, hist_valid, hist_abs, gt_abs)
                hist_abs = naomi_sample
                tensors_out[f'hist_abs_naomi'] = hist_abs

            if self.pretrained:
                with torch.no_grad():
                    if self.pretrained_module is not None:
                        sample = self.pretrained_module.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end)
                    elif (self.naomi) and self.pretrained_pred is not None:
                        sample = hist_abs
                    else:
                        sample = self.model.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end, hist_valid)
                # Don't care about the yaw differences
                new_hist_abs = sample[:, :, :2]
                # for start, _ in hist_seq_start_end:
                #     new_hist_abs[:, start] = hist_abs[:, start]
                for i in range(hist_abs.shape[1]):
                    if 'repl_all' in self.config.TRAJECTORY and self.config.TRAJECTORY.repl_all:
                        if (hist_valid[:, i] == 0).any():
                            continue
                    new_hist_abs[hist_valid[:, i] == 1, i] = hist_abs[hist_valid[:, i] == 1, i]
                hist_abs = new_hist_abs
                tensors_out[f'hist_abs_corr'] = hist_abs
            elif self.model.use_corr:
                with torch.no_grad():
                    sample = self.model.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end, hist_valid)
                # Don't care about the yaw differences
                new_hist_abs = sample[:, :, :2]
                for start, _ in hist_seq_start_end:
                    new_hist_abs[:, start] = hist_abs[:, start]
                hist_abs = new_hist_abs
                tensors_out[f'hist_abs_corr'] = hist_abs
           
            # For now, try after potential corrections
            if self.smoothing:
                hist_abs = processing.holt_winters_smoothing(hist_abs)
                tensors_out[f'hist_abs_smooth'] = hist_abs

            hist_rel = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_rel[1:] = hist_abs[1:] - hist_abs[:-1]
            num_batch = hist_rel.shape[1]

            hist_acc = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_acc[1:] = hist_rel[1:] - hist_rel[:-1]

            # 4d features
            hist_in = torch.cat([hist_abs, hist_rel, hist_acc], dim=-1)[..., -self.model.input_dim:]

            target_traj = torch.zeros((gt_abs.shape[1], hist_len, self.fut_len, self.dim), device=hist_abs.device)
            # Relative to last obs idx, cumulatively
            for obs_idx in range(hist_len):
                target_traj[:, obs_idx] = (gt_abs[obs_idx+1:obs_idx+1+self.fut_len] - gt_abs[obs_idx]).permute(1, 0, 2)
            
            # run forward propagation for the trajectory's history, assuming model is deterministic for warmup on obs
            with torch.no_grad():
                if self.pretrained_pred is not None:
                    all_goal_traj, all_dec_traj, KLD_loss, _ = \
                        self.pretrained_pred(hist_in, map_mask=None, targets=target_traj, start_index=0, training=False,
                                input_resnet=hist_resnet, seq_start_end=hist_seq_start_end)
                else:
                    all_goal_traj, all_dec_traj, KLD_loss, _ = \
                        self.model(hist_in, map_mask=None, targets=target_traj, start_index=0, training=False,
                                input_resnet=hist_resnet, seq_start_end=hist_seq_start_end)
                # Shapes: B x obs_len x pred_len x 2
                # convert the prediction to absolute coords
                # Shape = num_samples x N x B x d
                preds = hist_abs[-1] + all_dec_traj[:, -1].permute(2, 1, 0, 3)
                
            # compute best of num_samples
            best_idx = self.eval_metrics.update(gt_abs[hist_len:], preds, hist_seq_start_end)
            tensors_out[f'fut_abs'] = torch.stack([preds[best_idx_val, :, best_idx_i, :] \
                                                   for best_idx_i, best_idx_val in enumerate(best_idx)], dim=1)

            assert not self.visualize, 'Visualize not yet supported'
            # TODO: save outputs (?) Maybe in test only
            if self.visualize and batch_idx % self.plot_freq == 0:
                # self.generate_outputs(
                #     hist_abs, fut_abs, preds, best_sample_idx, seq_start_end, 
                #     f"epoch-{epoch+1}_val-{i}", epoch)
                pass
            self.save_tensors(tensors_out, batch_idx, epoch, 'test')
                        
        metrics = self.eval_metrics.get_metrics()
        self_name = self.out.base.split('/')[-1].split('_rel_2d')[0]
        print(f'{self_name} {time.strftime("%x %X")}: ADE={metrics["MinADE"]:.3f}, FDE={metrics["MinFDE"]:.3f}'\
                f', ADEMed={metrics["MinADEMed"]:.3f}, FDEMed={metrics["MinFDEMed"]:.3f}')
        return metrics

    def compute_loss(self, **kwargs) -> dict:
        """ Computes trainer's loss.
        Inputs:
        -------
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all computed losses. 
        """
        epoch = kwargs.get('epoch')
        goal = kwargs.get('goal')
        dec = kwargs.get('dec')
        kld = kwargs.get('kld')
        mse = kwargs.get('mse')
        return {
            'Loss': (goal+dec+kld+mse),
            'LossGoal': goal.item(), 
            'LossDec': dec.item(),
            'LossKLD': kld.item(),
            'LossMSE': mse.item()
        }

    def setup(self) -> None:
        """ Sets the trainer as follows:
            * model: SGNet_CVAE
            * optimizer: AdamW
            * lr_scheduler: ReduceOnPlateau 
        """
        logger.info(f"{self.name} setting up model: {self.trainer}")
    
        model_info = self.config.MODEL
        if 'pretrained' in self.config.BASE_CONFIG:
            paths = [self.config.BASE_CONFIG.pretrained['pretrained_module'], 
                     self.config.BASE_CONFIG.pretrained['pretrained_pred']]
            models = []
            if paths[0] and paths[1]:
                self.num_epoch = 1
            for pretrained_path in paths:
                if not pretrained_path:
                    models.append(None)
                    continue
                from run import get_exp_config
                pre_config = get_exp_config(pretrained_path, run_type='test', ckpt=None, fold=self.config.DATASET.name,
                                            gpu_id=self.config.BASE_CONFIG.gpu_id, use_cpu=self.config.BASE_CONFIG.use_cpu,
                                            max_test_epoch=self.config.BASE_CONFIG.max_test_epoch, corr=self.config.BASE_CONFIG.train_corr,
                                            epochs=self.config.BASE_CONFIG.n_epoch, no_tqdm=False)
                pre_config['load_ckpt'] = True
                pre_config['ckpt_name'] = False
                from vrnntools.trajpred_trainers.module import ModuleTrainer
                from vrnntools.trajpred_trainers.ego_vrnn import EgoVRNNTrainer
                from vrnntools.trajpred_trainers.ego_avrnn import EgoAVRNNTrainer
                #assert pre_config['trainer'] == 'module', 'Pretrained type must be module trainer'
                if pre_config['trainer'] == 'module':
                    trainer = ModuleTrainer(config=pre_config)
                elif pre_config['trainer'] == 'ego_vrnn':
                    trainer = EgoVRNNTrainer(config=pre_config)
                elif pre_config['trainer'] == 'ego_avrnn':
                    trainer = EgoAVRNNTrainer(config=pre_config)
                elif pre_config['trainer'] == 'sgnet':
                    trainer = SGNetCVAETrainer(config=pre_config)
                _ = trainer.eval(do_eval=False, load_only=True)
                models.append(trainer.model)
            self.pretrained = True
            self.pretrained_module = models[0]
            self.pretrained_pred = models[1]
        else:
            self.pretrained = False
            self.pretrained_module = None
            self.pretrained_pred = None

        self.model = SGNet_CVAE(model_info, self.device).to(self.device)
        if 'naomi' in self.config.TRAJECTORY and self.config.TRAJECTORY.naomi:
            self.naomi = True
            fold = self.config.DATASET.name
            from train import get_model
            naomi, _ = get_model(fold)
            naomi.eval()
            self.naomi_model = naomi
        else:
            self.naomi = False
        if self.naomi and self.pretrained_pred is not None:
            self.num_epoch = 1

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.TRAIN.lr)
        
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, threshold=1e-2, patience=10, factor=5e-1, 
            verbose=True)
        
        self.train_losses = metrics.LossContainer(loss_list=['Loss', 'LossGoal', 'LossDec', 'LossKLD', 'LossMSE'])
        # TODO: re-assess metrics, likely incorporate mAP (?)
        self.eval_metrics = MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE')

        if self.config.TRAIN.load_model:
            ckpt_file = os.path.join(self.out.ckpts, self.config.TRAIN.ckpt_name)
            assert os.path.exists(ckpt_file), \
                f"Checkpoint {ckpt_file} does not exist!"
            self.load_model(ckpt_file)
    
    def save_impl(self, epoch: int):
        pass

    def debug_impl(self):
        pass