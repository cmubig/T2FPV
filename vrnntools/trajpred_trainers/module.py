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

import vrnntools.utils.common as mutils

from vrnntools.trajpred_models.tp_module import ModuleRNN
from vrnntools.trajpred_trainers.base_trainer import BaseTrainer
from vrnntools.utils import metrics
from vrnntools.utils.metrics import LossContainer, MetricContainer
from vrnntools.utils.adj_matrix import ego_dists, simple_adjs, simple_distsim_adjs
#from vrnntools.utils.retracker import retrack
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class ModuleTrainer(BaseTrainer):
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
            
            hist_rel = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_rel[1:] = hist_abs[1:] - hist_abs[:-1]
            num_batch = hist_rel.shape[1]

            if self.model.use_corr:
                mse_corr = self.model.train_correction(gt_abs[:hist_len], gt_yaw[:hist_len], hist_abs, hist_yaw, hist_resnet[:hist_len], hist_seq_start_end)
            else:
                mse_corr = torch.autograd.Variable(torch.zeros((1,), device=hist_abs.device), requires_grad=True)

            loss = self.compute_loss(epoch=epoch, mse=mse_corr)
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

    def eval_epoch(self, epoch: int, **kwargs) -> dict:
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
        self.eval_metrics_orig.reset()
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

            self.eval_metrics_orig.update(gt_abs[:hist_len], hist_abs.unsqueeze(0), hist_seq_start_end)
            if self.model.use_corr:
                with torch.no_grad():
                    sample = self.model.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end)
                # Don't care about the yaw differences
                new_hist_abs = sample[:, :, :2]
                # for start, _ in hist_seq_start_end:
                #     new_hist_abs[:, start] = hist_abs[:, start]
                for i in range(hist_abs.shape[1]):
                    new_hist_abs[hist_valid[:, i] == 1, i] = hist_abs[hist_valid[:, i] == 1, i]
                hist_abs = new_hist_abs
           
            # compute best of num_samples
            # Shape = num_samples x N x B x d
            self.eval_metrics.update(gt_abs[:hist_len], hist_abs.unsqueeze(0), hist_seq_start_end)

            assert not self.visualize, 'Visualize not yet supported'
            # TODO: save outputs (?) Maybe in test only
            if self.visualize and i % self.plot_freq == 0:
                # self.generate_outputs(
                #     hist_abs, fut_abs, preds, best_sample_idx, seq_start_end, 
                #     f"epoch-{epoch+1}_val-{i}", epoch)
                pass
                        
        metrics = self.eval_metrics.get_metrics()
        orig_metrics = self.eval_metrics_orig.get_metrics()
        for k, v in orig_metrics.items():
            metrics[f'Orig{k}'] = v
        self_name = self.out.base.split('/')[-1].split('_rel_2d')[0]
        # print(f'{self_name} {time.strftime("%x %X")} Val: ADE={metrics["MinADE"]:.3f}, FDE={metrics["MinFDE"]:.3f}'\
        #         f', ADEMed={metrics["MinADEMed"]:.3f}, FDEMed={metrics["MinFDEMed"]:.3f}'\
        #         f', OrigADE={metrics["OrigMinADE"]:.3f}, OrigFDE={metrics["OrigMinFDE"]:.3f}'\
        #         f', OrigADEMed={metrics["OrigMinADEMed"]:.3f}, OrigFDEMed={metrics["OrigMinFDEMed"]:.3f}')
        print(f'{self_name} {time.strftime("%x %X")} Val: ADE={metrics["MinADE"]:.3f}, FDE={metrics["MinFDE"]:.3f}'\
                f', OrigADE={metrics["OrigMinADE"]:.3f}, OrigFDE={metrics["OrigMinFDE"]:.3f}')
        return metrics
    
    def test_epoch(self, epoch: int, **kwargs) -> dict:
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
        self.eval_metrics_orig.reset()
        self.cur_epoch = epoch
        self.cur_label = 'test'
        # self.eval_metrics_vel.reset()
        # self.eval_metrics_vel_orig.reset()
        # self.eval_metrics_acc.reset()
        # self.eval_metrics_acc_orig.reset()
        
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

            assert self.val_data.dataset.alignment, 'Unaligned data not supported'
            assert gt_valid.all(), 'Ground truth must be fully valid'
            assert not hist_all.isnan().any() and not hist_resnet.isnan().any(), 'NaN found in hist; interpolation needed?'
            assert (hist_seq_start_end == gt_seq_start_end).all(), 'Alignment must actually work...'

            # gt_vel = torch.zeros_like(gt_abs[:hist_len], device=gt_abs.device)
            # gt_acc = torch.zeros_like(gt_abs[:hist_len], device=gt_abs.device)
            # hist_vel = torch.zeros_like(gt_abs[:hist_len], device=gt_abs.device)
            # hist_acc = torch.zeros_like(gt_abs[:hist_len], device=gt_abs.device)
            # gt_vel[1:] = gt_abs[:hist_len][1:] - gt_abs[:hist_len][:-1]
            # gt_acc[1:] = gt_vel[1:] - gt_vel[:-1]
            # hist_vel[1:] = hist_abs[:hist_len][1:] - hist_abs[:hist_len][:-1]
            # hist_acc[1:] = hist_vel[1:] - hist_vel[:-1]
            tensors_out = {}
            tensors_out[f'seq_start_end'] = hist_seq_start_end
            tensors_out[f'hist_valid'] = hist_valid
            tensors_out[f'hist_abs'] = hist_abs
            tensors_out[f'hist_yaw'] = hist_yaw
            tensors_out[f'gt_abs'] = gt_abs
            tensors_out[f'gt_yaw'] = gt_yaw

            self.eval_metrics_orig.update(gt_abs[:hist_len], hist_abs.unsqueeze(0), hist_seq_start_end)
            # self.eval_metrics_vel_orig.update(gt_vel, hist_vel.unsqueeze(0), hist_seq_start_end)
            # self.eval_metrics_acc_orig.update(gt_acc, hist_acc.unsqueeze(0), hist_seq_start_end)
            if self.model.use_corr:
                with torch.no_grad():
                    sample = self.model.infer_correction(hist_abs, hist_yaw, hist_resnet, hist_seq_start_end)
                # Don't care about the yaw differences
                new_hist_abs = sample[:, :, :2]
                # for start, _ in hist_seq_start_end:
                #     new_hist_abs[:, start] = hist_abs[:, start]
                for i in range(hist_abs.shape[1]):
                    new_hist_abs[hist_valid[:, i] == 1, i] = hist_abs[hist_valid[:, i] == 1, i]
                hist_abs = new_hist_abs
                tensors_out[f'hist_abs_corr'] = hist_abs
           
            # compute best of num_samples
            # Shape = num_samples x N x B x d
            # hist_vel = torch.zeros_like(hist_abs[:hist_len], device=gt_abs.device)
            # hist_acc = torch.zeros_like(hist_abs[:hist_len], device=gt_abs.device)
            # hist_vel[1:] = hist_abs[:hist_len][1:] - hist_abs[:hist_len][:-1]
            # hist_acc[1:] = hist_vel[1:] - hist_vel[:-1]
            self.eval_metrics.update(gt_abs[:hist_len], hist_abs.unsqueeze(0), hist_seq_start_end)
            # self.eval_metrics_vel.update(gt_vel, hist_vel.unsqueeze(0), hist_seq_start_end)
            # self.eval_metrics_acc.update(gt_acc, hist_acc.unsqueeze(0), hist_seq_start_end)

            assert not self.visualize, 'Visualize not yet supported'
            # TODO: save outputs (?) Maybe in test only
            if self.visualize and batch_idx % self.plot_freq == 0:
                # self.generate_outputs(
                #     hist_abs, fut_abs, preds, best_sample_idx, seq_start_end, 
                #     f"epoch-{epoch+1}_val-{i}", epoch)
                pass
            self.save_tensors(tensors_out, batch_idx, epoch, 'test')
                        
        metrics = self.eval_metrics.get_metrics()
        orig_metrics = self.eval_metrics_orig.get_metrics()
        # vel_metrics = self.eval_metrics_vel.get_metrics()
        # vel_orig_metrics = self.eval_metrics_vel_orig.get_metrics()
        # acc_metrics = self.eval_metrics_acc.get_metrics()
        # acc_orig_metrics = self.eval_metrics_acc_orig.get_metrics()
        for k, v in orig_metrics.items():
            metrics[f'Orig{k}'] = v
        # for k, v in vel_metrics.items():
        #     metrics[f'Vel{k}'] = v
        # for k, v in vel_orig_metrics.items():
        #     metrics[f'VelOrig{k}'] = v
        # for k, v in acc_metrics.items():
        #     metrics[f'Acc{k}'] = v
        # for k, v in acc_orig_metrics.items():
        #     metrics[f'AccOrig{k}'] = v
        self_name = self.out.base.split('/')[-1].split('_rel_2d')[0]
        # print(f'{self_name} {time.strftime("%x %X")} Test: ADE={metrics["MinADE"]:.3f}, FDE={metrics["MinFDE"]:.3f}'\
        #         f', ADEMed={metrics["MinADEMed"]:.3f}, FDEMed={metrics["MinFDEMed"]:.3f}'\
        #         f', OrigADE={metrics["OrigMinADE"]:.3f}, OrigFDE={metrics["OrigMinFDE"]:.3f}'\
        #         f', OrigADEMed={metrics["OrigMinADEMed"]:.3f}, OrigFDEMed={metrics["OrigMinFDEMed"]:.3f}')
        #  print(f'{self_name} {time.strftime("%x %X")} Test: '\
        #       f'ADE={metrics["MinADE"]:.3f}, FDE={metrics["MinFDE"]:.3f} '\
        #       f'OrigADE={metrics["OrigMinADE"]:.3f}, OrigFDE={metrics["OrigMinFDE"]:.3f}, '\
        #       f'VelADE={metrics["VelMinADE"]:.3f}, VelFDE={metrics["VelMinFDE"]:.3f}, '\
        #       f'OrigVelADE={metrics["VelOrigMinADE"]:.3f}, OrigVelFDE={metrics["VelOrigMinFDE"]:.3f}, '\
        #       f'AccADE={metrics["AccMinADE"]:.3f}, AccFDE={metrics["AccMinFDE"]:.3f}, '\
        #       f'OrigAccADE={metrics["AccOrigMinADE"]:.3f}, OrigAccFDE={metrics["AccOrigMinFDE"]:.3f}'\
        # )
        print(f'{self_name} {time.strftime("%x %X")} Test: ADE={metrics["MinADE"]:.3f}, FDE={metrics["MinFDE"]:.3f}'\
                f', OrigADE={metrics["OrigMinADE"]:.3f}, OrigFDE={metrics["OrigMinFDE"]:.3f}')
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
        mse = kwargs.get('mse')
        return {
            'Loss': (mse),
            'LossCE': 0.0,
            'LossMSE': mse.item()
        }

    def setup(self) -> None:
        """ Sets the trainer as follows:
            * model: TrajPredVRNN
            * optimizer: AdamW
            * lr_scheduler: ReduceOnPlateau 
        """
        logger.info(f"{self.name} setting up model: {self.trainer}")
    
        model_info = self.config.MODEL

        self.model = ModuleRNN(model_info, self.device).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.TRAIN.lr)
        
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, threshold=1e-2, patience=10, factor=5e-1, 
            verbose=True)
        
        assert not self.train_corr, 'Use separate trainer for training corr'
        loss_list=['Loss', 'LossMSE']
        self.train_losses = LossContainer(loss_list=loss_list)
        # TODO: re-assess metrics, likely incorporate mAP (?)
        self.eval_metrics = MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE')
        self.eval_metrics_orig = MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE')
        # self.eval_metrics_vel = MetricContainer(metric_list=['MinADE', 'MinFDE'], main_metric='MinADE')
        # self.eval_metrics_vel_orig = MetricContainer(metric_list=['MinADE', 'MinFDE'], main_metric='MinADE')
        # self.eval_metrics_acc = MetricContainer(metric_list=['MinADE', 'MinFDE'], main_metric='MinADE')
        # self.eval_metrics_acc_orig = MetricContainer(metric_list=['MinADE', 'MinFDE'], main_metric='MinADE')

        if self.config.TRAIN.load_model:
            ckpt_file = os.path.join(self.out.ckpts, self.config.TRAIN.ckpt_name)
            assert os.path.exists(ckpt_file), \
                f"Checkpoint {ckpt_file} does not exist!"
            self.load_model(ckpt_file)
    
    def save_impl(self, epoch: int):
        pass