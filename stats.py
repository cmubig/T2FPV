import argparse
import os
import json
import numpy as np
import torch
import pandas as pd

from natsort import natsorted
from tqdm import tqdm

from vrnntools.utils.metrics import MetricContainer


def process_model(path, label):
    traj_dir = os.path.join(path, 'trajs')
    trajs = natsorted([os.path.join(traj_dir, x) for x in os.listdir(traj_dir)])
    # TODO: be careful about ensuring that all the batch files come from the same run!
    if not len(trajs):
        return None

    base_metrics = MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE')
    base_metrics.reset()
    ego_metrics = MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE')
    ego_metrics.reset()
    det_metrics = MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE')
    det_metrics.reset()

    det_thresholds = [torch.inf, 8, 4, 2, 1]
    all_det_metrics = [MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE') for _ in det_thresholds]
    det_totals = [0 for _ in det_thresholds]
    hist_len = 8
    total = 0
    egos = 0
    missing_det_points = 0

    hist_metrics = MetricContainer(metric_list=['MinADE', 'MinFDE', 'MinADEMed', 'MinFDEMed'], main_metric='MinADE')
    hist_metrics.reset()

    for batch_info in natsorted(trajs):
        batch = np.load(batch_info, allow_pickle=True).item()
        gt_abs, fut_abs = torch.from_numpy(batch['gt_abs']), torch.from_numpy(batch['fut_abs'])
        hist_abs = torch.from_numpy(batch['hist_abs'])
        hist_valid = torch.from_numpy(batch['hist_valid'])
        seq_start_end = torch.from_numpy(batch['seq_start_end'])

        ego_idxs = seq_start_end[:, 0].to(torch.long)
        det_idxs = torch.tensor([x for x in torch.arange(gt_abs.shape[1]) if x not in ego_idxs]).to(torch.long)
        base_metrics.update(gt_abs[8:], fut_abs.unsqueeze(0), seq_start_end)
        ego_metrics.update(gt_abs[8:][:, ego_idxs], fut_abs.unsqueeze(0)[:, :, ego_idxs], seq_start_end)
        if len(det_idxs):
            det_metrics.update(gt_abs[8:][:, det_idxs], fut_abs.unsqueeze(0)[:, :, det_idxs], seq_start_end)
        # TODO: move this computation stuff offline, into stats.py
        def do_thres(metric_container, threshold, gt_abs, preds, seq_start_end):
            # Apply thresholds
            all_err = torch.sqrt(torch.sum((do_thres.hist_abs.permute(1, 0, 2) - gt_abs[:hist_len].permute(1, 0, 2)) ** 2, dim=-1))
            ade_orig = torch.sum(all_err, dim=-1)/hist_len
            fde_orig = all_err[:, -1]
            init_orig = all_err[:, 0]
            threshold_filt = (ade_orig < threshold) 
            threshold_filt[seq_start_end[:, 0]] = False
            _ = metric_container.update(gt_abs[hist_len:][:, threshold_filt], preds[:, :, threshold_filt], None)
            return threshold_filt.sum().cpu().item()
        do_thres.hist_abs = torch.from_numpy(batch['hist_abs'])
        # Do thresholded computations
        for thres_idx, (threshold, container) in enumerate(zip(det_thresholds, all_det_metrics)):
            n_det = do_thres(container, threshold, gt_abs, fut_abs.unsqueeze(0), seq_start_end)
            det_totals[thres_idx] += n_det
        total += gt_abs.shape[1]
        egos += len(seq_start_end)
        # Make sure to only operate on dets here
        if len(det_idxs):
            hist_metrics.update(gt_abs[:8][:, det_idxs], hist_abs.unsqueeze(0)[:, :, det_idxs], seq_start_end)
            missing_det_points += (1 - hist_valid[:, det_idxs]).sum()

    metrics = base_metrics.get_metrics()
    metrics.update({f'Ego{k}': v for k, v in ego_metrics.get_metrics().items()})
    metrics.update({f'Det{k}': v for k, v in det_metrics.get_metrics().items()})
    metrics.update({f'DetHist{k}': v for k, v in hist_metrics.get_metrics().items()})
    for n_det, threshold, container in zip(det_totals, det_thresholds, all_det_metrics):
        thres_metrics = container.get_metrics()
        #print(f'\t{threshold} Thres: {n_det} dets, ADE={thres_metrics["MinADE"]:.3f}, FDE={thres_metrics["MinFDE"]:.3f}')
        thres_metrics['Dets'] = n_det
        metrics.update({f'DetThres{threshold}{k}': v for k, v in thres_metrics.items()})
    metrics['n_total'] = total
    metrics['n_ego'] = egos
    metrics['total_missing'] = missing_det_points
    metrics['missing_rate'] = missing_det_points / (8 * (total - egos))

    return metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path of either an out directory or exp_info json')
    args = parser.parse_args()
    path = args.path

    with open(path, 'r') as f:
        files = json.load(f)
    all_files = []
    all_items = []
    for item in files:
        if 'path' in item:
            if '*' in item['path']:
                #possible_folds = ['ETH', 'Hotel', 'Univ', 'Zara1', 'Zara2']
                possible_folds = ['ETH', 'Hotel', 'Univ', 'Zara1', 'Zara2']
                matching_paths = []
                folds = []
                for possible_fold in possible_folds:
                    path = item['path'].replace('*', possible_fold.lower())
                    if os.path.exists(path):
                        matching_paths.append(path)
                        folds.append(possible_fold)
            else:
                matching_paths = [item['path']]
                folds = [None]
            for fold, path in zip(folds, matching_paths):
                all_files.append((fold, path))
                all_items.append(item)

    all_res = []
    for (fold, path), item in tqdm(zip(all_files, all_items), 'Processing all files...', total=len(all_items), dynamic_ncols=True):
        fold_str = f'{fold}\t' if fold is not None else ''
        label = f'{fold_str}{item["key"]}'
        res = process_model(path, label)
        if res is None or not len(res):
            # File hasn't been created yet...
            continue
        res['fold'] = fold
        res['algo'] = item['key']
        res['train_name'] = item['train']
        all_res.append(res)
    all_res = pd.DataFrame(all_res).reset_index(drop=True)
    all_res = all_res[[x for x in all_res.columns if 'Med' not in x]]
    base_algos = []
    for i, row in all_res.iterrows():
        base_algos.append(row.algo.split(' ')[0])
    all_res['base_algo'] = base_algos

    # dfs = []
    # for i, group in all_res.groupby(['base_algo']):
    #     base_res = group[group.algo == group.iloc[0].algo].reset_index(drop=True)
    #     other_res = group[group.algo != group.iloc[0].algo].reset_index(drop=True)
    #     base_res['ade_diff'] = 1
    #     base_res['fde_diff'] = 1
    #     other_res['ade_diff'] = other_res.ade/base_res.ade.iloc[0]
    #     other_res['fde_diff'] = other_res.fde/base_res.fde.iloc[0]
    #     dfs.extend([base_res, other_res])

    avg_res = all_res.groupby(['algo', 'base_algo']).mean().reset_index()
    #print(avg_res[['algo', 'DetThresinfMinADE', 'DetThresinfMinFDE', 'DetThresinfDets']])
    print(avg_res[['algo', 'MinADE', 'MinFDE']])
    tot = all_res.groupby('fold')['n_total'].mean()
    ego = all_res.groupby('fold')['n_ego'].mean()
    det = all_res.groupby('fold')['DetThresinfDets'].mean()
    sus = pd.DataFrame({'tot': tot, 'ego': ego, 'det': det})
    sus['per_ego'] = sus.det / sus.ego
    # 1 value per fold
    sus['total_missing'] = all_res.groupby('fold')['total_missing'].mean()
    sus['missing_rate'] = all_res.groupby('fold')['missing_rate'].mean()
    sus['Hist MSE'] = all_res.groupby('fold')['DetHistMinADE'].mean()
    import pdb; pdb.set_trace()

