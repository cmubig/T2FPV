import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from natsort import natsorted
import glob
import pickle as pkl
from p_tqdm import p_map

output_base = './data/FPVDataset/mp4'
cache_path = './data/input_data.pkl'

dataset_folds = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
input_folds = ['biwi_eth', 'biwi_hotel', 'students001', 'students003',
               'uni_examples', 'crowds_zara01', 'crowds_zara02', 'crowds_zara03']

input_loc = {'biwi_eth': 'eth', 'biwi_hotel': 'hotel', 'students001': 'univ', 'students003': 'univ',
             'crowds_zara01': 'zara', 'crowds_zara02': 'zara', 'crowds_zara03': 'zara', 'uni_examples': 'univ'}



def get_input_data(cache=False, cache_path=cache_path):
    if os.path.exists(cache_path) and cache:
        with open(cache_path, 'rb') as f:
            to_cache = pkl.load(f)
        return to_cache['ret'], to_cache['detection_dict']
    all_folds = []
    
    def process_fold(fold):
        fold_path = os.path.join(output_base, fold)
        csv_files = natsorted(glob.glob(os.path.join(fold_path, 'agent*_path.csv')))
        meta_files = natsorted(glob.glob(os.path.join(fold_path, 'agent*_meta.csv')))
        assert len(csv_files) == len(meta_files), 'Mismatch between paths and meta files'
        fold_data = []
        for csv_file in csv_files:
            agent_id = int(csv_file.split('agent')[-1].split('_')[0])
            data = pd.read_csv(csv_file, sep=', ', engine='python')
            data['agent_id'] = agent_id
            fold_data.append(data)
        fold_data = pd.concat(fold_data).sort_values(['frame_id', 'agent_id']).reset_index(drop=True)
        # reference_fold = os.path.join(split_path, 'processed', f'{fold}.csv')
        # reference_data = pd.read_csv(reference_fold, header=None)
        # assert len(reference_data) == len(fold_data), 'Mismatch between reference and fold data sizes'
        fold_data['scene'] = fold
        new_cols = [fold_data.columns[-1], fold_data.columns[-2], *fold_data.columns[:-2]]
        fold_data = fold_data[new_cols]
        return fold_data

    # for fold in tqdm(input_folds, 'Loading folds...', total=len(input_folds), dynamic_ncols=True):
    #     fold_data = process_fold(fold)
    #     all_folds.append(fold_data)
    all_folds = p_map(process_fold, input_folds, desc='Loading folds...')

    ret = pd.concat(all_folds).sort_values(['scene', 'agent_id', 'frame_id']).reset_index(drop=True)
    
    detection_dict = {}
    col_names = ['det_id', 'x_w', 'y_w', 'z_w', 'x_c', 'y_c', 'z_c',
                    'yaw_w', 'yaw_c', 'width_w', 'height_w', 'depth_w',
                    'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'num_pixels']
    col_names_str = ','.join([f'"{x}"' for x in col_names])
    # For each agent, what agents they see, and for what frames...
    for _, row in tqdm(ret.iterrows(), 'Processing det_ids', total=len(ret), dynamic_ncols=True):
        scene_dict = detection_dict.setdefault(row.scene, {})
        agent_dict = scene_dict.setdefault(row.agent_id, {})
        #det_str = f'[[{col_names_str}],{row.det_ids[1:]}'
        det_str = eval(row.det_ids)
        # if row.det_ids != '[]':
        #     dets = eval(row.det_ids)
        #     dets = pd.DataFrame(dets)
        #     dets.columns = col_names
        # else:
        #     dets = pd.DataFrame(columns=col_names)
        
        agent_dict[row.frame_id] = det_str
    with open(cache_path, 'wb') as f:
        to_cache = {'ret': ret, 'detection_dict': detection_dict}
        pkl.dump(to_cache, f)
    return ret, detection_dict

if __name__ == '__main__':
    # First need to load in all the data
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()

    data, detection_dict = get_input_data(cache=args.cache)
