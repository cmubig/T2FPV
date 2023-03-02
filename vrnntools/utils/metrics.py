# ------------------------------------------------------------------------------
# @file:    metrics.py
# @brief:   This file contains the implementation of the measure and loss classes
#           class which for computing and containing all metrics needed for 
#           evaluation. 
# @author:  Ingrid Navarro, Ben Stoler
# @date:    Last modified on August 3rd, 2022
# ------------------------------------------------------------------------------

import torch
import numpy as np
import pandas as pd
import math

class LossContainer:
    """ A class for maintaining losses """
    def __init__(self, loss_list = ['Loss', 'LossKLD', 'LossNLL']) -> None:
        # initialize 
        self.losses = dict()
        for loss in loss_list:
            self.losses[loss] = Loss()
            
    def reset(self) -> None:
        for k, v in self.losses.items():
            v.reset()
            
    def update(self, losses, batch_size):
        for loss in losses:
            for k, v in loss.items():
                v = v.item() if torch.is_tensor(v) else v
                if not self.losses.get(k):
                    continue
                self.losses[k].compute(batch_size=batch_size, value=v)
    
    def get_losses(self) -> dict:
        
        losses = {}
        for k, v in self.losses.items():
            losses[k] = v.get_metric()
        return losses

class MetricContainer:
    """ A class for maintaining metric """
    def __init__(
        self, metric_list = ['MinADE', 'MinFDE', 'ADE', 'FDE'], main_metric = 'MinADE') -> None:
        
        assert main_metric in metric_list, f"{main_metric} not in {metric_list}"

        self.main_metric = main_metric
        
        # initialize 
        self.metrics = dict()
        for metric in metric_list:
            if metric == 'ADE':
                self.metrics[metric] = ADE(mode='mean')
            elif metric == 'MinADE':
                self.metrics[metric] = ADE(mode='min')
            elif  metric == 'FDE':
                self.metrics[metric] = FDE(mode='mean')
            elif metric == 'MinFDE':
                self.metrics[metric] = FDE(mode='min')
            elif metric == 'MinADEMed':
                self.metrics[metric] = ADEMed(mode='min')
            elif metric == 'MinFDEMed':
                self.metrics[metric] = FDEMed(mode='min')
            elif metric == 'ThresMinADE':
                self.metrics[metric] = ThresADE(mode='min')
            elif metric == 'ThresADE':
                self.metrics[metric] = ThresADE(mode='mean')
            else:
                raise NotImplementedError(f"Metric {metric} not supported!")

    def reset(self) -> None:
        
        for k, v in self.metrics.items():
            v.reset()
    
    def update(self, traj_gt, traj_pred, seq_start_end, permute=True, scene_metrics=False):
        
        idx = self.metrics[self.main_metric].compute(
            traj_gt=traj_gt, traj_pred=traj_pred, seq_start_end=seq_start_end, 
            permute=permute, scene_metrics=scene_metrics)

        for k, v in self.metrics.items(): 
            if k == self.main_metric:
                continue
            v.compute(
                traj_gt=traj_gt, traj_pred=traj_pred, seq_start_end=seq_start_end, 
                permute=permute, scene_metrics=scene_metrics)
            
        return idx
            
    def get_metrics(self) -> dict:
        metrics = {}
        for k, v in self.metrics.items():
            m = v.get_metric()
            # One level deep of nested metrics supported
            if hasattr(m, 'items'):
                for k2, v2 in m.items():
                    v2 = v2.item() if torch.is_tensor(v2) else v2
                    metrics[k2] = v2
            else:
                m = m.item() if torch.is_tensor(m) else m
                metrics[k] = m
        return metrics
    
class Metric:
    """ Base class for implementing metrics. """
    def __init__(self) -> None:
        """ Initialization. """
        self.metric = 0.0
        self.accum = 0
    
    def reset(self, value = 0.0) -> float:
        self.metric = value
        self.accum = 0
        
    def get_metric(self) -> float:
        if self.accum < 1:
            return 0.0
        return self.metric / self.accum
    
    def compute(self, **kwargs) -> float:
        raise NotImplementedError
    
class Loss:
    """ Base class for implementing losses. """
    def __init__(self) -> None:
        """ Initialization. """
        self.metric = 0.0
        self.accum = 0
    
    def reset(self, value = 0.0) -> float:
        self.metric = value
        self.accum = 0
        
    def get_metric(self) -> float:
        if self.accum < 1:
            return 0.0
        return self.metric / self.accum
    
    def compute(self, **kwargs) -> float:
        self.accum += kwargs.get('batch_size')
        self.metric += kwargs.get('value')
        
class ADE(Metric):
    """ Computes min/mean average displacement error (ADE) of N trajectories. """
    def __init__(self, mode = 'mean') -> None:
        super().__init__()
        
        self.mode = mode
        
    def compute(self, **kwargs):
        permute = True
        if not kwargs.get('permute') is None:
            permute = kwargs.get('permute')
        
        # NOTE:
        # traj_gt -> (1, pred_len, batch_size, dim)
        # traj_pred -> (num_samples, pred_len, batch_size, dim)
        
        traj_gt = kwargs.get('traj_gt').unsqueeze(0)
        traj_pred = kwargs.get('traj_pred')
        seq_start_end = kwargs.get('seq_start_end')
        scene_metrics = kwargs.get('scene_metrics')
        
        assert traj_gt[0].shape == traj_pred[0].shape, \
            f"Shape mismatch: gt {traj_gt.shape} pred {traj_pred.shape}"
        
        seq_len, batch_size, _ = traj_pred[0].shape
        
        if permute:
            # y1 -> (1, batch_size, pred_len, dim)
            y1 = traj_gt.permute(0, 2, 1, 3)
            # y2 -> (num_samples, batch_size, pred_len, dim)
            y2 = traj_pred.permute(0, 2, 1, 3)
        
        # ade -> (num_samples, batch_size)
        ade = torch.nansum(torch.sqrt(torch.sum((y1 - y2) ** 2, dim=3)), dim=2)
        if scene_metrics:
            assert False, 'Scene metrics not implemented'
        else:
            idx = None  
            if self.mode == 'min':
                ade_min = torch.min(ade, dim=0)
                ade, idx = ade_min.values, ade_min.indices
            if self.mode == 'mean':
                ade_mean = torch.mean(ade, dim=0)
                ade = ade_mean
            self.metric += torch.sum(ade)

        self.accum += batch_size * seq_len
        return idx
        
class ADEMed(Metric):
    """ Computes median of min/mean average displacement error (ADE) of N trajectories. """
    def __init__(self, mode = 'mean') -> None:
        super().__init__()
        
        self.mode = mode
        self.all_vals = []
    
    def reset(self, value=0.0):
        super().reset(value=value)
        self.all_vals = []
    
    def get_metric(self):
        return np.median(np.array(self.all_vals))
        
    def compute(self, **kwargs):
        permute = True
        if not kwargs.get('permute') is None:
            permute = kwargs.get('permute')
        
        # NOTE:
        # traj_gt -> (1, pred_len, batch_size, dim)
        # traj_pred -> (num_samples, pred_len, batch_size, dim)
        
        traj_gt = kwargs.get('traj_gt').unsqueeze(0)
        traj_pred = kwargs.get('traj_pred')
        seq_start_end = kwargs.get('seq_start_end')
        scene_metrics = kwargs.get('scene_metrics')
        
        assert traj_gt[0].shape == traj_pred[0].shape, \
            f"Shape mismatch: gt {traj_gt.shape} pred {traj_pred.shape}"
        
        seq_len, batch_size, _ = traj_pred[0].shape
        
        if permute:
            # y1 -> (1, batch_size, pred_len, dim)
            y1 = traj_gt.permute(0, 2, 1, 3)
            # y2 -> (num_samples, batch_size, pred_len, dim)
            y2 = traj_pred.permute(0, 2, 1, 3)
        
        # ade -> (num_samples, batch_size)
        ade = torch.nansum(torch.sqrt(torch.sum((y1 - y2) ** 2, dim=3)), dim=2)
        if scene_metrics:
            assert False, 'Scene metrics not implemented'
        else:
            idx = None  
            if self.mode == 'min':
                ade_min = torch.min(ade, dim=0)
                ade, idx = ade_min.values, ade_min.indices
            if self.mode == 'mean':
                ade_mean = torch.mean(ade, dim=0)
                ade = ade_mean
            self.all_vals.extend(list(ade.cpu().numpy()/seq_len))
            self.metric += torch.sum(ade)

        self.accum += batch_size * seq_len
        return idx
    
class FDE(Metric):
    """ Computes average displacement error (ADE). """
    def __init__(self, mode = 'mean') -> None:
        super().__init__()
        
        self.mode = mode
        
    def compute(self, **kwargs):
        """ Computes and accumulates final displacement error (fde). 
        Inputs:
        ------
        endpoint_gt[torch.tensor]: ground truth final positions with shape (batch, dim).
        endpoint_pred[torch.tensor]: predicted final positions with shape (batch, dim).
        """
        endpoint_gt = kwargs.get('traj_gt').unsqueeze(0)[:, -1]
        endpoint_pred = kwargs.get('traj_pred')[:, -1]
        seq_start_end = kwargs.get('seq_start_end')
        scene_metrics = kwargs.get('scene_metrics')
        
        assert endpoint_gt[0].shape == endpoint_pred[0].shape, \
            f"Shape mismatch: gt {endpoint_gt.shape} pred {endpoint_pred.shape}"
        
        _, batch_size, _ = endpoint_gt.shape
        
        fde = torch.sqrt(torch.nansum((endpoint_gt - endpoint_pred) ** 2, dim=2))
        if scene_metrics:
            assert False, 'Scene metrics not implemented'
        else:
            if self.mode == 'mean':
                fde = torch.mean(fde, dim=0)
            elif self.mode == 'min': 
                fde = torch.min(fde, dim=0).values
            self.metric += torch.sum(fde)
    
        self.accum += batch_size


class FDEMed(Metric):
    """ Computes medians average displacement error (ADE). """
    def __init__(self, mode = 'mean') -> None:
        super().__init__()
        
        self.mode = mode
        self.all_vals = []
    
    def reset(self, value=0.0):
        super().reset(value=value)
        self.all_vals = []
    
    def get_metric(self):
        return np.median(np.array(self.all_vals))
        
    def compute(self, **kwargs):
        """ Computes and accumulates final displacement error (fde). 
        Inputs:
        ------
        endpoint_gt[torch.tensor]: ground truth final positions with shape (batch, dim).
        endpoint_pred[torch.tensor]: predicted final positions with shape (batch, dim).
        """
        endpoint_gt = kwargs.get('traj_gt').unsqueeze(0)[:, -1]
        endpoint_pred = kwargs.get('traj_pred')[:, -1]
        seq_start_end = kwargs.get('seq_start_end')
        scene_metrics = kwargs.get('scene_metrics')
        
        assert endpoint_gt[0].shape == endpoint_pred[0].shape, \
            f"Shape mismatch: gt {endpoint_gt.shape} pred {endpoint_pred.shape}"
        
        _, batch_size, _ = endpoint_gt.shape
        
        fde = torch.sqrt(torch.nansum((endpoint_gt - endpoint_pred) ** 2, dim=2))
        if scene_metrics:
            assert False, 'Scene metrics not implemented'
        else:
            if self.mode == 'mean':
                fde = torch.mean(fde, dim=0)
            elif self.mode == 'min': 
                fde = torch.min(fde, dim=0).values
            self.all_vals.extend(list(fde.cpu().numpy()))
            self.metric += torch.sum(fde)
    
        self.accum += batch_size


class ThresADE(Metric):
    """ Computes Thresholded min/mean average displacement error (ADE) of N trajectories. """
    def __init__(self, mode = 'mean') -> None:
        super().__init__()
        
        self.mode = mode
        assert self.mode == 'min', 'Only min mode supported currently'
        self.thresholds = np.linspace(0.5, 3, 6)
        self.threshold_idx = 1 # threshold ADE of 1
        assert self.thresholds[self.threshold_idx] == 1.0, 'Mismatch in consistent threshold'
        self.recall_levels = np.linspace(0, 1, 101)
        self.init_accums()
    
    def reset(self, value=0.0):
        super().reset(value=value)
        self.init_accums()

    def init_accums(self):
        self.threshold_accum = [{'ego_ades': [], 'ego_fdes': [],
                                 'match_labels': [], 'match_ades': [], 'match_fdes': [], 'match_num_gt': 0} \
                                for _ in range(len(self.thresholds))]
    
    def get_metric(self):
        # From here: https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/ 
        APs = []
        # Only use detections/matches for calculating Precsion/Recall
        for info in self.threshold_accum:
            sort_idxs = np.argsort(info['match_ades'])
            labels = np.array(info['match_labels'])[sort_idxs]
            ades = np.array(info['match_ades'])[sort_idxs]
            cum_tp = 0
            cum_fp = 0
            tp_rows = []
            fp_rows = []
            for label in labels:
                if label == 'tp':
                    cum_tp += 1
                else:
                    cum_fp += 1
                tp_rows.append(cum_tp)
                fp_rows.append(cum_fp)
            df = pd.DataFrame({'ade': ades, 
                               'label': labels,
                               'cum_tp': tp_rows, 
                               'cum_fp': fp_rows,
                               'num_gt': info['match_num_gt']})
            df['precision'] = df.cum_tp / (df.cum_fp + df.cum_tp)
            df['recall'] = df.cum_tp / df.num_gt
            prec_tot = 0
            for recall_level in self.recall_levels:
                prec = df[df.recall > recall_level].precision.max()
                if math.isnan(prec):
                    prec = 0
                prec_tot += prec
            AP = prec_tot/len(self.recall_levels)
            APs.append(AP)


        # TODO: handle ego separately
        info = self.threshold_accum[self.threshold_idx]
        ade = np.mean(np.concatenate([info['match_ades'], info['ego_ades']]))
        match_ade = np.mean(info['match_ades'])
        ego_ade = np.mean(info['ego_ades'])

        fde = np.mean(np.concatenate([info['match_fdes'], info['ego_fdes']]))
        match_fde = np.mean(info['match_fdes'])
        ego_fde = np.mean(info['ego_fdes'])
        AP = APs[self.threshold_idx]
        mAP = np.mean(APs)
        return {'ThresMinADE': ade, 'ThresMinFDE': fde, 'AP1': AP, 'mAP': mAP, 
                'EgoADE': ego_ade, 'EgoFDE': ego_fde,
                'DetADE': match_ade, 'DetFDE': match_fde}
        
    def compute(self, **kwargs):
        permute = True
        if not kwargs.get('permute') is None:
            permute = kwargs.get('permute')
        
        # NOTE:
        # traj_gt -> (1, pred_len, batch_size, dim)
        # traj_pred -> (num_samples, pred_len, batch_size, dim)
        
        traj_gts = kwargs.get('traj_gt').unsqueeze(0)
        traj_preds = kwargs.get('traj_pred')
        seq_start_end = kwargs.get('seq_start_end')
        obs_seq_start_end, pred_seq_start_end = seq_start_end

        for obs_start_end, pred_start_end in zip(obs_seq_start_end, pred_seq_start_end):
            traj_pred = traj_preds[:, :, obs_start_end[0]:obs_start_end[1], :]
            traj_gt = traj_gts[:, :, pred_start_end[0]:pred_start_end[1], :]
            if permute:
                # y1 -> (1, batch_size, pred_len, dim)
                traj_gt = traj_gt.permute(0, 2, 1, 3)
                # y2 -> (num_samples, batch_size, pred_len, dim)
                traj_pred = traj_pred.permute(0, 2, 1, 3)
            gt_size = traj_gt.shape[1]

            # TODO: Use all vs. any here?
            # For now, only use full ground truths (for ETH, removes 87% of predictions....)
            traj_gt_filtered = [traj_gt[:, i, ...] for i in range(gt_size) if not traj_gt[:, i, ...].isnan().any()]
            assert len(traj_gt_filtered), 'At least one GT for Ego must be provided'
            traj_gt_filtered = torch.stack(traj_gt_filtered, dim=1)

            filtered_gt_size = traj_gt_filtered.shape[1]
            traj_gt_lens = torch.cat([traj_gt.shape[2] - traj_gt_filtered[:, i, :, 0].isnan().sum(dim=-1) \
                                         for i in range(filtered_gt_size)])

            # Fut traj errors across (obs_samples x obs_dets x gt_dets)
            # Ensure that ADE takes into consideration the length
            # Shape is obs by gt
            abs_err = torch.sum(torch.sqrt(torch.nansum((traj_gt_filtered - traj_pred.unsqueeze(2)) ** 2, dim=-1)), dim=-1)
            abs_err = abs_err / traj_gt_lens
            abs_err_final = torch.sqrt(torch.nansum((traj_gt_filtered - traj_pred.unsqueeze(2)) ** 2, dim=-1))[..., -1]
            for idx, traj_gt_len in enumerate(traj_gt_lens):
                if traj_gt_len != traj_gt.shape[2]:
                    abs_err_final[..., idx] = np.inf

            abs_err_mins,_  = abs_err.min(dim=0)
            abs_err_mins_idxs = abs_err.argmin(dim=0, keepdim=True)
            abs_err_final_mins = torch.gather(abs_err_final, dim=0, index=abs_err_mins_idxs)[0]
            # Handling ego separately
            ego_ade = abs_err_mins[0, 0]
            ego_fde = abs_err_final_mins[0, 0]
            abs_err_mins = abs_err_mins[1:, 1:]
            abs_err_final_mins = abs_err_final_mins[1:, 1:]

            match_ades = []
            match_fdes = []
            while abs_err_mins.shape[0] > 0 and abs_err_mins.shape[1] > 0:
                min_pair = abs_err_mins.min()
                min_idx = abs_err_mins.argmin()
                obs_idx = torch.div(min_idx, abs_err_mins.shape[1], rounding_mode='floor')
                pred_idx = min_idx % abs_err_mins.shape[1]
                match_ades.append(min_pair)
                match_fde = abs_err_final_mins[obs_idx, pred_idx]
                if not torch.isinf(match_fde):
                    match_fdes.append(match_fde)

                new_obs_idxs = torch.cat([torch.arange(0, obs_idx), torch.arange(obs_idx+1, abs_err_mins.shape[0])])
                new_pred_idxs = torch.cat([torch.arange(0, pred_idx), torch.arange(pred_idx+1, abs_err_mins.shape[1])])
                abs_err_mins = abs_err_mins[torch.meshgrid(new_obs_idxs, new_pred_idxs)]
                abs_err_final_mins = abs_err_final_mins[torch.meshgrid(new_obs_idxs, new_pred_idxs)]

            for thresh_idx, threshold in enumerate(self.thresholds):
                self.threshold_accum[thresh_idx]['ego_ades'].append(ego_ade.item())
                self.threshold_accum[thresh_idx]['ego_fdes'].append(ego_fde.item())

                self.threshold_accum[thresh_idx]['match_labels'].extend([('fp' if x >= threshold else 'tp') for x in match_ades])
                self.threshold_accum[thresh_idx]['match_num_gt'] += (filtered_gt_size - 1)
                self.threshold_accum[thresh_idx]['match_ades'].extend([x.item() for x in match_ades])
                self.threshold_accum[thresh_idx]['match_fdes'].extend([x.item() for x in match_fdes])


def thres_rmse(x_pred, x_true, obs_seq_start_end, pred_seq_start_end, assume_aligned=False):
    batch_ades = []
    if assume_aligned:
        assert not x_pred.isnan().any() and not x_true.isnan().any(), 'Mismatch in alignment assumption'
        assert len(x_pred) == len(x_true), 'Mismatch in alignment assumption (len)'
        abs_err = torch.sum(torch.sqrt(torch.nansum((x_pred - x_true) ** 2, dim=-1)), dim=-1).mean(dim=-1) 
        return abs_err
    for obs_start_end, pred_start_end in zip(obs_seq_start_end, pred_seq_start_end):
        traj_pred = x_pred[obs_start_end[0]:obs_start_end[1], :]
        traj_gt = x_true[ pred_start_end[0]:pred_start_end[1], :]

        gt_size = len(traj_gt)
        # Consistency, only train on fully non-nan preds
        traj_gt_filtered = [traj_gt[i, ...] for i in range(gt_size) if not traj_gt[i, -1].isnan().any()]
        if len(traj_gt_filtered):
            traj_gt = torch.stack(traj_gt_filtered)
        else:
            traj_gt = torch.zeros((0, *traj_gt.shape[1:]), device=traj_gt.device)
        # average along each pred
        abs_err = torch.sum(torch.sqrt(torch.nansum((traj_gt - traj_pred.unsqueeze(1)) ** 2, dim=-1)), dim=-1).mean(dim=-1)
        match_ades = []
        while abs_err.shape[0] > 0 and abs_err.shape[1] > 0:
            if assume_aligned:
                min_pair = abs_err[0, 0]
                obs_idx, pred_idx = 0, 0
            else:
                min_pair = abs_err.min()
                min_idx = abs_err.argmin()
                obs_idx = min_idx // abs_err.shape[1]
                pred_idx = min_idx % abs_err.shape[1]
            match_ades.append(min_pair)

            new_obs_idxs = torch.cat([torch.arange(0, obs_idx), torch.arange(obs_idx+1, abs_err.shape[0])])
            new_pred_idxs = torch.cat([torch.arange(0, pred_idx), torch.arange(pred_idx+1, abs_err.shape[1])])
            abs_err = abs_err[torch.meshgrid(new_obs_idxs, new_pred_idxs)]
        match_ades = torch.stack(match_ades)
        batch_ades.append(match_ades)
        
    return torch.cat(batch_ades)

def thres_cvae(x_pred, x_true, obs_seq_start_end, pred_seq_start_end, assume_aligned=False):
    batch_ades = []
    if assume_aligned:
        assert not x_pred.isnan().any() and not x_true.isnan().any(), 'Mismatch in alignment assumption (nan)'
        assert len(x_pred) == len(x_true), 'Mismatch in alignment assumption (len)'
        x_pred = x_pred.permute(3, 0, 1, 2, 4)
        abs_err, _ = torch.sum(torch.sqrt(torch.nansum((x_pred - x_true) ** 2, dim=-1)), dim=-1).mean(dim=-1).min(dim=0)
        return abs_err
    for obs_start_end, pred_start_end in zip(obs_seq_start_end, pred_seq_start_end):
        traj_pred = x_pred[obs_start_end[0]:obs_start_end[1], :]
        traj_gt = x_true[ pred_start_end[0]:pred_start_end[1], :]

        gt_size = len(traj_gt)
        # Consistency, only train on fully non-nan preds
        traj_gt_filtered = [traj_gt[i, ...] for i in range(gt_size) if not traj_gt[i, -1].isnan().any()]
        if len(traj_gt_filtered):
            traj_gt = torch.stack(traj_gt_filtered)
        else:
            traj_gt = torch.zeros((0, *traj_gt.shape[1:]), device=traj_gt.device)
        # average along each pred
        traj_pred = traj_pred.permute(3, 0, 1, 2, 4)
        abs_err, _ = torch.sum(torch.sqrt(torch.nansum((traj_gt - traj_pred.unsqueeze(2)) ** 2, dim=-1)), dim=-1).mean(dim=-1).min(dim=0)
        match_ades = []
        while abs_err.shape[0] > 0 and abs_err.shape[1] > 0:
            if assume_aligned:
                min_pair = abs_err[0, 0]
                obs_idx, pred_idx = 0, 0
            else:
                min_pair = abs_err.min()
                min_idx = abs_err.argmin()
                obs_idx = min_idx // abs_err.shape[1]
                pred_idx = min_idx % abs_err.shape[1]
            match_ades.append(min_pair)

            new_obs_idxs = torch.cat([torch.arange(0, obs_idx), torch.arange(obs_idx+1, abs_err.shape[0])])
            new_pred_idxs = torch.cat([torch.arange(0, pred_idx), torch.arange(pred_idx+1, abs_err.shape[1])])
            abs_err = abs_err[torch.meshgrid(new_obs_idxs, new_pred_idxs)]
        match_ades = torch.stack(match_ades)
        batch_ades.append(match_ades)
        
    return torch.cat(batch_ades)