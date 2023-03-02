# ------------------------------------------------------------------------------
# @file:    vrnn.py
# @brief:   This file contains the implementation of the VRNNTrainer class which
#           is used for training a Variational Recurrent Neural Network for 
#           trajectory prediction. 
# @author:  Ingrid Navarro, Ben Stoler
# @date:    Last modified on August 25th, 2022
# ------------------------------------------------------------------------------
import logging
import os
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm

import vrnntools.utils.common as mutils

from vrnntools.trajpred_models.tp_vrnn import TrajPredVRNN
from vrnntools.trajpred_trainers.base_trainer import BaseTrainer
from vrnntools.utils import metrics
from vrnntools.utils.adj_matrix import ego_dists, simple_adjs, simple_distsim_adjs

logger = logging.getLogger(__name__)

class VRNNTrainer(BaseTrainer):
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
        
    def train_epoch(self, epoch: int) -> dict:
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

        for i, batch in tqdm(enumerate(self.train_data), dynamic_ncols=True, desc=epoch_str, total=self.num_iter):
            if i >= self.num_iter:
                break
            
            self.optimizer.zero_grad()

            batch = [tensor.to(self.device) for tensor in batch]   
            hist_abs, hist_resnet, hist_seq_start_end, fut_abs, fut_seq_start_end = batch
            assert not hist_abs.isnan().any() and not hist_resnet.isnan().any(), 'Unexpected NaN in hist'
            hist_abs, hist_yaw, hist_abs_img, hist_valid = \
                hist_abs[..., :2], hist_abs[..., 2], hist_abs[..., 3:5], hist_abs[..., 5]
            gt_abs, gt_yaw, gt_abs_img, gt_valid = \
                fut_abs[..., :2], fut_abs[..., 2], fut_abs[..., 3:5], fut_abs[..., 5]
            hist_len = hist_abs.shape[0]
            fut_abs, fut_yaw, fut_abs_img, fut_valid = gt_abs[hist_len:], gt_yaw[hist_len:], gt_abs_img[hist_len:], gt_valid[hist_len:]

            assert (hist_seq_start_end == fut_seq_start_end).all(), 'In training, alignment required'
            # Only fully observed gt_abs stuff...
            if self.interp_valid_only:
                # Mask out incomplete ones, in train only
                valid_mask = torch.all(gt_valid > 0, dim=0)
                hist_resnet = hist_resnet[:, valid_mask]
                hist_abs = hist_abs[:, valid_mask]
                hist_yaw = hist_yaw[:, valid_mask]
                hist_abs_img = hist_abs_img[:, valid_mask]
                gt_abs = gt_abs[:, valid_mask]
                gt_yaw = gt_yaw[:, valid_mask]
                gt_abs_img = gt_abs_img[:, valid_mask]
                fut_abs, fut_yaw, fut_abs_img = gt_abs[hist_len:], gt_yaw[hist_len:], gt_abs_img[hist_len:]
                len_filt = []
                for start, end in hist_seq_start_end:
                    len_filt.append(valid_mask[start:end].sum().item())
                seq_idx_filt = [0] + np.cumsum(len_filt).tolist()
                hist_seq_start_end = np.array([[start, end] for start, end in zip(seq_idx_filt, seq_idx_filt[1:])])
                hist_seq_start_end = torch.from_numpy(hist_seq_start_end).to(hist_abs.device)
                fut_seq_start_end = hist_seq_start_end.clone()
            
            hist_rel = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_rel[1:] = hist_abs[1:] - hist_abs[:-1]
            # if self.config.BASE_CONFIG['model_design']['feat_enc_x']['in_size'] == 4:
            #     hist_acc = torch.zeros(hist_abs.shape).to(hist_abs.device)
            #     hist_acc[1:] = hist_rel[1:] - hist_rel[:-1]
            #     hist_rel = torch.cat([hist_rel, hist_acc], dim=-1)
            num_batch = hist_rel.shape[1]

            # Compute adjacency matrices...
            seq_adj = simple_adjs(hist_abs[0].unsqueeze(0), hist_seq_start_end)[0]
            hist_adj = simple_distsim_adjs(hist_abs, hist_seq_start_end, self.model.sigma, seq_adj=seq_adj)

            # only relative coordinates are supported
            kld, nll, _ = self.model(hist_rel, hist_resnet, hist_abs, hist_seq_start_end, hist_adj)
                
            loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll)
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
        
        self.eval_losses.reset()
        self.eval_metrics.reset()
        
        epoch_str = f"[{epoch}/{self.num_epoch}] Val"
        for i, batch in tqdm(enumerate(self.val_data), dynamic_ncols=True, desc=epoch_str, total=self.eval_num_iter):
            if i >= self.eval_num_iter:
                break

            batch = [tensor.to(self.device) for tensor in batch]   
            hist_abs, hist_resnet, hist_seq_start_end, fut_abs, fut_seq_start_end = batch
            assert not hist_abs.isnan().any() and not hist_resnet.isnan().any(), 'Unexpected NaN in hist'
            hist_abs, hist_yaw, hist_abs_img, hist_valid = \
                hist_abs[..., :2], hist_abs[..., 2], hist_abs[..., 3:5], hist_abs[..., 5]
            gt_abs, gt_yaw, gt_abs_img, gt_valid = \
                fut_abs[..., :2], fut_abs[..., 2], fut_abs[..., 3:5], fut_abs[..., 5]
            hist_len = hist_abs.shape[0]
            fut_abs, fut_yaw, fut_abs_img, fut_valid = gt_abs[hist_len:], gt_yaw[hist_len:], gt_abs_img[hist_len:], gt_valid[hist_len:]

            hist_rel = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_rel[1:] = hist_abs[1:] - hist_abs[:-1]
            # if self.config.BASE_CONFIG['model_design']['feat_enc_x']['in_size'] == 4:
            #     hist_acc = torch.zeros(hist_abs.shape).to(hist_abs.device)
            #     hist_acc[1:] = hist_rel[1:] - hist_rel[:-1]
            #     hist_rel = torch.cat([hist_rel, hist_acc], dim=-1)
            num_batch = hist_rel.shape[1]

            # Compute adjacency matrices...
            seq_adj = simple_adjs(hist_abs[0].unsqueeze(0), hist_seq_start_end)[0]
            hist_adj = simple_distsim_adjs(hist_abs, hist_seq_start_end, self.model.sigma, seq_adj=seq_adj)

            
            loss_list = []
            pred_list = []
            # run forward propagation for the trajectory's history, assuming model is deterministic for warmup on obs
            with torch.no_grad():
                kld, nll, h_ = self.model(hist_rel, hist_resnet, hist_abs, hist_seq_start_end, hist_adj)
            for _ in range(num_samples):
                h = h_.clone()

                # run inference to predict the trajectory's future steps
                pred_rel = self.model.inference(self.fut_len, h, hist_abs, hist_seq_start_end)
                # convert the prediction to absolute coords
                if self.coord != 'abs':
                    pred = mutils.convert_rel_to_abs(
                        pred_rel, hist_abs[-1], permute=True)
                else:
                    pred = pred_rel
                pred_list.append(pred)
                
                loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll)
                loss_list.append(loss)
                
            # compute best of num_samples
            # Shape = num_samples x N x B x d
            preds = torch.stack(pred_list)
            # Stay in pred_rel mode
            # Need to do precision/recall aware prediction..., thresholded
            
            seq_start_end = (hist_seq_start_end, fut_seq_start_end)
            best_sample_idx = self.eval_metrics.update(fut_abs, preds, seq_start_end, scene_metrics=self.scene_metrics)

            assert not self.visualize, 'Visualize not yet supported'
            if self.visualize and i % self.plot_freq == 0:
                # self.generate_outputs(
                #     hist_abs, fut_abs, preds, best_sample_idx, seq_start_end, 
                #     f"epoch-{epoch+1}_val-{i}", epoch)
                pass
                        
        metrics = self.eval_metrics.get_metrics()
        print({k: float(f'{v:.5f}') for k, v in metrics.items()})
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
        
        self.eval_losses.reset()
        self.eval_metrics.reset()
        
        epoch_str = f"[{epoch}/{self.num_epoch}] Test"
        for i, batch in tqdm(enumerate(self.test_data), dynamic_ncols=True, desc=epoch_str, total=self.test_num_iter):
            if i >= self.test_num_iter:
                break
            
            batch = [tensor.to(self.device) for tensor in batch]   
            hist_abs, hist_resnet, hist_seq_start_end, fut_abs, fut_seq_start_end = batch
            assert not hist_abs.isnan().any() and not hist_resnet.isnan().any(), 'Unexpected NaN in hist'
            hist_abs, hist_yaw, hist_abs_img, hist_valid = \
                hist_abs[..., :2], hist_abs[..., 2], hist_abs[..., 3:5], hist_abs[..., 5]
            gt_abs, gt_yaw, gt_abs_img, gt_valid = \
                fut_abs[..., :2], fut_abs[..., 2], fut_abs[..., 3:5], fut_abs[..., 5]
            hist_len = hist_abs.shape[0]
            fut_abs, fut_yaw, fut_abs_img, fut_valid = gt_abs[hist_len:], gt_yaw[hist_len:], gt_abs_img[hist_len:], gt_valid[hist_len:]

            hist_rel = torch.zeros(hist_abs.shape).to(hist_abs.device)
            hist_rel[1:] = hist_abs[1:] - hist_abs[:-1]
            # if self.config.BASE_CONFIG['model_design']['feat_enc_x']['in_size'] == 4:
            #     hist_acc = torch.zeros(hist_abs.shape).to(hist_abs.device)
            #     hist_acc[1:] = hist_rel[1:] - hist_rel[:-1]
            #     hist_rel = torch.cat([hist_rel, hist_acc], dim=-1)
            num_batch = hist_rel.shape[1]

            # Compute adjacency matrices...
            seq_adj = simple_adjs(hist_abs[0].unsqueeze(0), hist_seq_start_end)[0]
            hist_adj = simple_distsim_adjs(hist_abs, hist_seq_start_end, self.model.sigma, seq_adj=seq_adj)

            loss_list = []
            pred_list = []
            # run forward propagation for the trajectory's history
            with torch.no_grad():
                kld, nll, h_ = self.model(hist_rel, hist_resnet, hist_abs, hist_seq_start_end, hist_adj)
            for _ in range(num_samples):
                h = h_.clone()

                # run inference to predict the trajectory's future steps
                pred_rel = self.model.inference(self.fut_len, h, hist_abs, hist_seq_start_end)
                
                # convert the prediction to absolute coords
                if self.coord != 'abs':
                    pred = mutils.convert_rel_to_abs(
                        pred_rel, hist_abs[-1], permute=True)
                else:
                    pred = pred_rel
                pred_list.append(pred)
                
                loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll)
                loss_list.append(loss)
                
            # compute best of num_samples
            preds = torch.stack(pred_list)
            seq_start_end = (hist_seq_start_end, fut_seq_start_end)
            best_sample_idx = self.eval_metrics.update(fut_abs, preds, seq_start_end, scene_metrics=self.scene_metrics)
    
            if self.visualize and i % self.plot_freq == 0:
                # self.generate_outputs(
                #     hist_abs, fut_abs, preds, best_sample_idx, seq_start_end, 
                #     f"epoch-{epoch+1}_val-{i}", epoch)
                pass
                        
        metrics = self.eval_metrics.get_metrics()
        print({k: float(f'{v:.5f}') for k, v in metrics.items()})
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
        kld = kwargs.get('kld')
        nll = kwargs.get('nll')
        return {
            'Loss': (self.warmup[epoch-1] * kld + nll),
            'LossKLD': kld.item(), 
            'LossNLL': nll.item(),
            'LossCE': 0.0,
            'LossMSE': 0.0
        }

    def setup(self) -> None:
        """ Sets the trainer as follows:
            * model: TrajPredVRNN
            * optimizer: AdamW
            * lr_scheduler: ReduceOnPlateau 
        """
        logger.info(f"{self.name} setting up model: {self.trainer}")
    
        model_info = self.config.MODEL

        self.model = TrajPredVRNN(model_info, self.device).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.TRAIN.lr)
        
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, threshold=1e-2, patience=10, factor=5e-1, 
            verbose=True)
        
        self.train_losses = metrics.LossContainer()
        self.eval_losses = metrics.LossContainer()
        # TODO: re-assess metrics, but for now we will see
        self.eval_metrics = metrics.MetricContainer(metric_list=['ThresMinADE'], main_metric='ThresMinADE')

        if self.config.TRAIN.load_model:
            ckpt_file = os.path.join(self.out.ckpts, self.config.TRAIN.ckpt_name)
            assert os.path.exists(ckpt_file), \
                f"Checkpoint {ckpt_file} does not exist!"
            self.load_model(ckpt_file)
    
    def save_impl(self, epoch: int):
        pass