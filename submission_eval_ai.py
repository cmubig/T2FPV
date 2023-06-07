import argparse
import json
import os
import numpy as np

from natsort import natsorted
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-config', default='./config/fpv_det_train/sgnet_cvae.json', type=str,
        help='path to experiment configuration file')
    parser.add_argument('--output', default='submission_test.npy', type=str, help='Output path for submission')
    
    args = parser.parse_args()

    with open(args.exp_config, 'r') as f:
        config = json.load(f)
    base_config = config['base_config']
    with open(base_config, 'r') as f:
        base_config = json.load(f)
    model_config = config['model_design']
    with open(model_config, 'r') as f:
        model_config = json.load(f)

    folds = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    traj_config = base_config['trajectory']
    out_paths = []
    out_epochs = []
    for fold in folds:
        exp_tag = config['exp_tag'].replace('eth', fold)
        exp_name = "{}_{}_{}_{}d_hl-{}_hs-{}_fl-{}_fs-{}".format(
                    exp_tag, 
                    config['trainer'], 
                    config['coord'], 
                    model_config['dim'], 
                    traj_config['hist_len'], 
                    traj_config['hist_step'], 
                    traj_config['fut_len'], 
                    traj_config['fut_step'],
                )
        out_path = f'out/{fold}/{exp_name}'
        out_paths.append(out_path)
        trainval_path = out_path + '/trainval.log'
        with open(trainval_path, 'r') as f:
            lines = f.readlines()
        run_markers = []
        for i, line in enumerate(lines):
            if 'epoch: [0' in line:
                run_markers.append(i)
        assert len(run_markers), 'Could not find previous run'
        cur_run = lines[run_markers[-1]:]

        cur_epoch = 0
        best_epoch = -1
        best_ade = np.inf
        for line in cur_run:
            if 'Epoch' in line:
                cur_epoch = int(line.split('[')[-1].split('/')[0])
            if 'eval: ' not in line:
                continue
            ade = float(line.split(' MinADE: ')[-1].split(' ')[0])
            if ade < best_ade:
                best_epoch = cur_epoch
                best_ade = ade
                best_line = line
        assert best_epoch > 0, 'Could not find best epoch to load from'
        out_epochs.append(best_epoch)
    
    pred_out = {fold: [] for fold in folds}
    for fold, out_path, epoch in zip(folds, out_paths, out_epochs):
        print(fold)
        all_batches = os.listdir(f'{out_path}/trajs')
        all_batches = [x for x in all_batches if x.startswith(f'test_epoch{epoch}_batch')]
        all_batches = natsorted(all_batches)
        for batch_idx, batch in tqdm(enumerate(all_batches), total=len(all_batches)):
            batch = np.load(f'{out_path}/trajs/{batch}', allow_pickle=True).item()
            pred_out[fold].append(batch['fut_abs'])
    np.save(args.output, pred_out)
        
