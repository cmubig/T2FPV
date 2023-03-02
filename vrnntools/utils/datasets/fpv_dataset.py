import pandas as pd
import numpy as np
import glob
import logging
import torch
import cv2
import time
import sys
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from natsort import natsorted
from tqdm import tqdm
from p_tqdm import p_map, t_map
from scipy import interpolate
from scipy.optimize import linear_sum_assignment

import vrnntools.utils.common as mutils
from vrnntools.utils.cache_fpv import get_input_data, input_loc

logger = logging.getLogger(__name__)

def filtered_glob(path):
    return [x for x in natsorted(glob.glob(path)) if not x.endswith('.meta')]

class FPVDataset:

    def __init__(self, label, label_file_end, traj_config, data_config):
        self.label = label
        self.label_file_end = label_file_end
        self.load_images = data_config['load_images']
        self.load_detections = data_config['load_detections']
        self.alignment = data_config['alignment']
        self.hungarian_match = 'hungarian_match' in data_config and data_config['hungarian_match']
        if self.hungarian_match:
            assert self.alignment, 'Hungarian match creates alignment'
        self.load_resnet = data_config['load_resnet']

        self.splits_path = data_config['splits_path']
        self.dets_path = data_config['dets_path']
        self.imgs_path = data_config['imgs_path']
        self.dataset_fold = data_config['name']

        self.img_width = data_config['img_width']
        self.img_height = data_config['img_height']
        self.vis_ratio = data_config['vis_ratio']

        self.hist_len = traj_config['hist_len']        
        self.hist_step = traj_config['hist_step']        
        self.fut_len = traj_config['fut_len']        
        self.fut_step = traj_config['fut_step']        
        self.balance = data_config['balance']['enabled'] if 'balance' in data_config else False
        self.balance_seed = data_config['balance']['seed'] if self.balance else 1
        self.balance_factor = data_config['balance']['factors'][label] if self.balance else 1
        self.balance_max = data_config['balance']['max_scene'] if self.balance else 1000000

        self.min_detection_len = data_config['min_detection_len']
        assert self.hist_step == self.fut_step, 'Hist and fut step must match'

        self.full_len = self.hist_len + self.fut_len
        # Number of steps to skip over, when partitioning scenes
        self.step_skip = traj_config['skip']
        # Number of frames to skip per step
        self.frame_skip = traj_config['frame_skip']
        assert self.frame_skip == 10, 'Only frame_skip of 10 supported for imgs'

        self.min_agents = traj_config['min_agents']

        # Used to assess predictions & det/track info
        if FPVDataset.cached_input_data is None:
            data_gt, detection_dict_gt = get_input_data(cache=data_config['use_input_cache'],
                                                cache_path=data_config['input_cache_path'])
            FPVDataset.cached_input_data = (data_gt, detection_dict_gt)
        else:
            data_gt, detection_dict_gt = FPVDataset.cached_input_data
        
        self.noise = None
        self.noise_random = None
        if 'noise' in data_config:
            noise_config = data_config['noise']
            if noise_config['enabled']:
                self.noise = noise_config
                self.noise_random = np.random.RandomState(self.noise['seed'])

        scene_ids = []
        cur_scene_id = 0

        # Different seq lens for obs & pred, depending on detection performance (precision, etc.)
        all_obs = None
        all_pred = None
        all_img_paths = None
        seq_len_obs = []
        seq_len_pred = []

        class dynamic_array:
            def __init__(self, buffer_size=1024):
                self.entry_buffer = []
                self.buffer_size = buffer_size
                self.finalized = None
            
            def accumulate(self, entry):
                self.entry_buffer.append(entry)
                if len(self.entry_buffer) >= self.buffer_size:
                    self.finalize()
            
            def finalize(self):
                finalized_buffer = np.concatenate(self.entry_buffer)
                if self.finalized is None:
                    self.finalized = finalized_buffer
                else:
                    self.finalized = np.concatenate([self.finalized, finalized_buffer])
                self.entry_buffer = []
                return self.finalized
        
        def concat_init(obj, new_entry):
            if obj is None:
                obj = dynamic_array()
            obj.accumulate(new_entry)
            return obj

        input_files = filtered_glob(os.path.join(self.splits_path, self.dataset_fold, self.label, '*.txt'))
        meta_df = self._balance_locs(input_files, self.balance)

        for input_file in tqdm(input_files, f'Processing {self.label} input...', dynamic_ncols=True):
            file_data = pd.read_csv(input_file, header=None, delim_whitespace=True)
            file_data.columns = ['frame_id', 'agent_id', 'y', 'x']
            file_data = file_data[['frame_id', 'agent_id', 'x', 'y']]
            file_data.frame_id = file_data.frame_id.astype(int)
            file_data.agent_id = file_data.agent_id.astype(int)

            full_name = input_file.split('/')[-1].replace(self.label_file_end, '')
            full_gt = data_gt[data_gt.scene == full_name]
            full_dict_gt = detection_dict_gt[full_name]

            min_frame = file_data.frame_id.min()
            max_frame = file_data.frame_id.max()

            # As in Social GAN, overlapping sliding window with specified skip
            # Filter by number of min agents required (+1 more, since they use strict greater than operator)
            # Agents which have partial tracks within a potential scene are thrown out
            for frame_start in tqdm(range(min_frame, max_frame + 1, self.step_skip*self.frame_skip),
                                    f'Processing scenes in {input_file.split("/")[-1]}', dynamic_ncols=True, leave=False):
                scene_data = file_data[(file_data.frame_id >= frame_start) &\
                                       (file_data.frame_id < frame_start + self.full_len*self.frame_skip)]
                scene_data = scene_data.groupby('agent_id').filter(lambda x: len(x) == self.full_len)
                n_agents = scene_data.agent_id.nunique()
                # Greater than or equal to, rather than strict greater than as in others
                if not n_agents >= self.min_agents:
                    continue
                if not len(meta_df[(meta_df.file == full_name) & (meta_df.frame_start == frame_start)]):
                    continue
                cur_scene_id += 1

                def process_agent(agent_id, agent_df):
                    nonlocal all_obs
                    nonlocal all_pred
                    nonlocal all_img_paths
                    
                    # Scene full must be true
                    obs_frames = agent_df[['frame_id']].values[:self.hist_len].squeeze()
                    pred_frames = agent_df[['frame_id']].values[self.hist_len:].squeeze()
                    # Shape is full_len x 2
                    assert len(obs_frames) + len(pred_frames) == self.full_len, 'Mismatch in track shape'
                    ego_track = agent_df[['x', 'y']].values
                    ego_ref = full_gt[(full_gt.agent_id == agent_id) &\
                                    ((full_gt.frame_id.isin(obs_frames)) | (full_gt.frame_id.isin(pred_frames)))]
                    assert len(ego_ref) == len(ego_track), 'Mismatch in ego ref to track'
                    # Add in orient & val columns: [x, y, orient, img_x, img_y, valid]
                    ego_img = np.zeros((ego_track.shape[0], 2))
                    ego_img[:, 0] = self.img_width/2
                    ego_img[:, 1] = self.img_height/2
                    ego_track = np.concatenate([ego_track, ego_ref.yaw.values[:, np.newaxis], ego_img], axis=-1)
                    ego_obs = np.concatenate([ego_track[:self.hist_len], 
                                              np.ones((self.hist_len,1), int),
                                              np.ones((self.hist_len,1), int)*agent_id], axis=-1)
                    ego_pred = np.concatenate([ego_track[:], 
                                              np.ones((self.fut_len+self.hist_len,1),int),
                                              np.ones((self.fut_len+self.hist_len,1),int)*agent_id], axis=-1)

                    if self.load_detections:
                        det_obs, det_pred, gt_ids = self._get_detections(
                            full_name, full_gt, full_dict_gt, agent_id, frame_start, obs_frames, pred_frames
                        )

                        all_obs = concat_init(all_obs, np.concatenate([ego_obs[np.newaxis, ...], det_obs]))
                        all_pred = concat_init(all_pred, np.concatenate([ego_pred[np.newaxis, ...], det_pred]))
                        seq_len_obs.append(len(det_obs) + 1)
                        seq_len_pred.append(len(det_pred) + 1)
                    else:
                        gt_ids = []
                        all_obs = concat_init(all_obs, ego_obs[np.newaxis, ...])
                        all_pred = concat_init(all_pred, ego_pred[np.newaxis, ...])
                        seq_len_obs.append(1)
                        seq_len_pred.append(1)
                    
                    # Just store the paths, then do one bulk update of images/agents themselves?
                    if self.load_images:
                        # every 10 frames (self.frame_skip) from  the first frame
                        # Make sure to also get detections' agent_ids + future path as well...
                        #combined_ids = [agent_id, *gt_ids]
                        combined_ids = [agent_id]
                        for idx, combined_id in enumerate(combined_ids):
                            if idx == 0 or self.alignment:
                                combined_start_frame = file_data[file_data.agent_id == combined_id].frame_id.min()
                                combined_offset = (frame_start - combined_start_frame)//self.frame_skip
                                img_folder = os.path.join(self.imgs_path, full_name, f'agent{combined_id}')
                                #img_files = [os.path.join(img_folder, f'idx{combined_offset + i}.jpg') for i in range(self.full_len)]
                                img_files = [os.path.join(img_folder, f'idx{combined_offset + i}.jpg') for i in range(self.hist_len)]
                            else:
                                # Load as nan's later
                                combined_offset = "BAD_PATH"
                                img_folder = os.path.join(self.imgs_path, full_name, f'agent{combined_id}')
                                #img_files = [os.path.join(img_folder, f'{combined_offset}.jpg') for i in range(self.full_len)]
                                img_files = [os.path.join(img_folder, f'{combined_offset}.jpg') for i in range(self.hist_len)]
                            imgs = np.array(img_files)
                            all_img_paths = concat_init(all_img_paths, imgs[np.newaxis, ...])
                    else:
                        imgs = np.array([None] * self.hist_len)
                        all_img_paths = concat_init(all_img_paths, imgs[np.newaxis, ...])
                        pass

                    scene_ids.append(cur_scene_id)

                for agent_id, agent_df in scene_data.groupby('agent_id'):
                    process_agent(agent_id, agent_df)

        if self.load_images:
            all_img_paths = all_img_paths.finalize()
            img_paths_needed = np.array(list(set(all_img_paths.flatten())))
            # TODO: Other visual features: Poses? Depth estimations? Semantic segmentation? Instance segmentation?
            # e.g. https://pytorch.org/vision/0.13/models.html#semantic-segmentation
            # DeepLabv3 with ResNet50 backbone!
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            resnet_pre = ResNet50_Weights.IMAGENET1K_V2.transforms()
            res_modules=list(resnet.children())[:-1]
            resnet=torch.nn.Sequential(*res_modules)
            for p in resnet.parameters():
                p.requires_grad = False
            resnet.eval()
            res_device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
            resnet = resnet.to(res_device)

            def get_img(path):
                if not os.path.exists(path):
                    return torch.zeros((3, 480, 640))*np.nan
                img = Image.open(path).convert('RGB')
                # Desired: C x H x W
                return torch.from_numpy(np.transpose(np.asarray(img, np.float32) / 255.0, axes=[2, 0, 1]))

            # imgs = []
            features = []
            buffer = []
            buffer_cnt = 0
            def conv_buffer():
                nonlocal buffer
                nonlocal buffer_cnt
                if buffer_cnt == 0:
                    return
                buffer_cnt = 0
                # imgs.append(resnet_pre(torch.stack(buffer)))
                # features.append(resnet(imgs[-1].to(res_device)).squeeze(-1).squeeze(-1).cpu().detach())
                buffer_stack = torch.stack(buffer)
                if buffer_stack.isnan().any():
                    import pdb; pdb.set_trace()
                img_pre = resnet_pre(buffer_stack)
                resnet_out = resnet(img_pre.to(res_device)).squeeze(-1).squeeze(-1).cpu().detach()
                features.append(resnet_out)
                buffer = []

            for i, path in enumerate(tqdm(img_paths_needed, desc='Loading images to cache...', dynamic_ncols=True)):
                buffer_cnt += 1
                buffer.append(get_img(path))
                if buffer_cnt >= 128:
                    conv_buffer()
            conv_buffer()
            # self.imgs = torch.cat(imgs)
            # imgs = []
            self.features = torch.cat(features) 
            features = []
            self.img_cache = {path: i for i, path in enumerate(img_paths_needed)}
        else:
            all_img_paths = all_img_paths.finalize()
            self.features = torch.from_numpy(np.zeros((1, 2048)).astype(np.float32))
            self.img_cache = {None: 0}

        self.all_img_paths = all_img_paths

        seq_len_obs = np.array(seq_len_obs).astype(int)
        seq_len_pred = np.array(seq_len_pred).astype(int)
        scene_ids = np.array(scene_ids).astype(int)

        all_obs = all_obs.finalize().astype(np.float32)
        all_pred = all_pred.finalize().astype(np.float32)


        # Update all_ego_resnet if appropriate
        #num_valid = sum([1 for x in self.features if not x.isnan().any()])
        self.all_resnet = torch.zeros((*self.all_img_paths.shape, 2048))
        if self.load_images:
            for i in range(len(self.all_resnet)):
                paths = self.all_img_paths[i]
                i_features = torch.stack([self.features[self.img_cache[path]] for path in paths])
                self.all_resnet[i] = i_features
            # TODO: maybe keep these around idk
            self.features = None
            self.img_cache = None
            self.all_img_paths = None

        # CONVENTION: first entry is always ego
        self.all_obs = torch.from_numpy(all_obs)
        self.all_pred = torch.from_numpy(all_pred)
        self.scene_ids = torch.from_numpy(scene_ids)

        self.seq_len_obs = torch.from_numpy(seq_len_obs)
        self.seq_len_pred = torch.from_numpy(seq_len_pred)

        obs_cum_start_idx = [0] + np.cumsum(seq_len_obs).tolist()
        obs_seq_start_end = np.array([[start, end] for start, end in zip(obs_cum_start_idx, obs_cum_start_idx[1:])])

        pred_cum_start_idx = [0] + np.cumsum(seq_len_pred).tolist()
        pred_seq_start_end = np.array([[start, end] for start, end in zip(pred_cum_start_idx, pred_cum_start_idx[1:])])
        self.obs_seq_start_end = torch.from_numpy(obs_seq_start_end)
        self.pred_seq_start_end = torch.from_numpy(pred_seq_start_end)

    def __len__(self):
        return len(self.obs_seq_start_end)
        # return len(self.scene_ids.unique())

    def __getitem__(self, idx):

        def get_agent_info(agent_idx):
            def get_idxs(seq_start_end):
                return torch.from_numpy(np.array([i for i in range(seq_start_end[agent_idx, 0], 
                                                                   seq_start_end[agent_idx, 1])]))
            obs_idxs = get_idxs(self.obs_seq_start_end)
            pred_idxs = get_idxs(self.pred_seq_start_end)

            traj_obs = self.all_obs[obs_idxs]
            traj_pred = self.all_pred[pred_idxs]
            #resnet_gt = self.all_resnet[obs_idxs]
            resnet_gt = self.all_resnet[agent_idx]
            return [traj_obs, traj_pred, resnet_gt]
        #return [get_agent_info(agent_idx) for agent_idx in agent_idxs]
        return get_agent_info(idx)

    def _balance_locs(self, input_files, balance=False):
        # Determining how many "scenes" there will be
        n_scenes = 0
        scene_locs = []
        scene_files = []
        frame_starts = []
        scene_agents = []
        for input_file in tqdm(input_files, f'Counting {self.label} input...', dynamic_ncols=True):
            file_data = pd.read_csv(input_file, header=None, delim_whitespace=True)
            file_data.columns = ['frame_id', 'agent_id', 'y', 'x']
            file_data = file_data[['frame_id', 'agent_id', 'x', 'y']]
            file_data.frame_id = file_data.frame_id.astype(int)
            full_name = input_file.split('/')[-1].replace(self.label_file_end, '')
            min_frame = file_data.frame_id.min()
            max_frame = file_data.frame_id.max()
            for frame_start in tqdm(range(min_frame, max_frame + 1, self.step_skip*self.frame_skip),
                                    f'Counting scenes in {input_file.split("/")[-1]}', dynamic_ncols=True, leave=False):
                scene_data = file_data[(file_data.frame_id >= frame_start) &\
                                        (file_data.frame_id < frame_start + self.full_len*self.frame_skip)]
                scene_data = scene_data.groupby('agent_id').filter(lambda x: len(x) == self.full_len)
                n_agents = scene_data.agent_id.nunique()
                # Greater than or equal to, rather than strict greater than as in others
                if not n_agents >= self.min_agents:
                    continue
                n_scenes += n_agents
                scene_agents.append(n_agents)
                frame_starts.append(frame_start)
                scene_locs.append(input_loc[full_name])
                scene_files.append(full_name)
        meta_df = pd.DataFrame({'locs': scene_locs, 'file': scene_files, 'frame_start': frame_starts, 'n_agents': scene_agents})
        # No balancing
        if not balance:
            return meta_df
        
        tot_agents = meta_df.n_agents.sum()
        desired_agents = min(self.balance_max, tot_agents * self.balance_factor)
        loc_agents = meta_df.groupby('locs').n_agents.sum().sort_values()
        loc_desired = {}
        for i, (loc, loc_avail) in enumerate(loc_agents.iteritems()):
            balance_floor = desired_agents // (meta_df.locs.nunique() - i)
            loc_desired[loc] = min(balance_floor, loc_avail)
            desired_agents -= loc_desired[loc]

        tmp_df = meta_df.copy()
        tmp_df['orig_idx'] = tmp_df.index
        bal_random = np.random.RandomState(self.balance_seed)
        vals = tmp_df.values
        bal_random.shuffle(vals)
        shuf_df = pd.DataFrame.from_records(vals, columns=tmp_df.columns)

        keep = []
        for _, row in shuf_df.iterrows():
            if loc_desired[row.locs] >= 0:
                loc_desired[row.locs] -= row.n_agents
                keep.append(True)
            else:
                keep.append(False)
        shuf_df['keep'] = keep
        shuf_df = shuf_df[shuf_df.keep]
        meta_df = meta_df.iloc[shuf_df.orig_idx].sort_index().reset_index(drop=True)
        return meta_df


    def _get_detections(self, full_name, full_gt, full_dict_gt, agent_id, frame_start, obs_frames, pred_frames):
        # Now use self.dets_path to get obs; full_gt will be the ground truth to compare against
        #dets_file = os.path.join(self.dets_path, full_name, f'agent{agent_id}_dets.csv')
        dets_folder = os.path.join(self.dets_path, full_name)
        if os.path.exists(dets_folder):
            dets_file = [x for x in os.listdir(dets_folder) if f'agent{agent_id}_' in x]
            assert len(dets_file) == 1, 'Could not find corresponding det file'
            dets_file = os.path.join(dets_folder, dets_file[0])

            dets_df = pd.read_csv(dets_file)
            dets_df = dets_df[(dets_df.scene_start == frame_start) &\
                            (dets_df.frame_id % self.frame_skip == 0)]
        else:
            # TODO: @Soonmin, do D&T for crowds_zara03 and uni_examples in sgan_split
            base_columns = ['scene_start', 'frame_id', 'det_id', 'x_w', 'y_w', 'z_w', 'x_c', 'y_c',
                            'z_c', 'yaw_w', 'yaw_c', 'width_w', 'height_w', 'depth_w', 'bbox_left',
                            'bbox_top', 'bbox_width', 'bbox_height']
            dets_df = pd.DataFrame([], columns = base_columns)
        dets_df['original_id']  = dets_df['det_id']
        # Now apply noise if needed!
        if self.noise is not None:
            # First check odds of completely dropping trajectory
            assigned_ids = dets_df.det_id.unique()
            drop_odds = 1.0/self.noise['drop_traj']
            to_drop = set([x for x in assigned_ids if self.noise_random.random() < drop_odds])
            dets_df = dets_df[~dets_df.det_id.isin(to_drop)]
            # Now, process per frame per agent
            if len(dets_df):
                next_agent_id = assigned_ids.max() + 1
            else:
                next_agent_id = None
            for assigned_id, noise_agent_df in dets_df.groupby('det_id'):
                assert next_agent_id is not None, 'None check failed'
                frame_odds = 1.0/self.noise['drop_frame']
                lose_odds = 1.0/self.noise['lose_traj']
                cur_id = assigned_id
                indices_to_drop = []
                for row_idx in noise_agent_df.index:
                    if self.noise_random.random() < frame_odds:
                        indices_to_drop.append(row_idx)
                    if self.noise_random.random() < lose_odds:
                        cur_id = next_agent_id
                        next_agent_id += 1
                    row = dets_df.loc[row_idx]
                    noise_x = self.noise['mu'] + self.noise['std']*self.noise_random.randn() + row.x_w
                    noise_z = self.noise['mu'] + self.noise['std']*self.noise_random.randn() + row.z_w
                    # Increase noise for pixel bbox...i.e. from 0.05m to 5 pixels
                    noise_left = self.noise['mu'] + 100*self.noise['std']*self.noise_random.randn() + row.bbox_left
                    noise_top = self.noise['mu'] + 100*self.noise['std']*self.noise_random.randn() + row.bbox_top
                    dets_df.loc[row_idx, ['det_id', 'x_w', 'z_w', 'bbox_left', 'bbox_top']] = \
                        [cur_id, noise_x, noise_z, noise_left, noise_top] 
                if len(indices_to_drop):
                    dets_df = dets_df.drop(index=indices_to_drop)
        # Go through assigned_ids now, if partial mask with nan
        det_obs_dict = {'frame_id': obs_frames}
        for det_agent, det_agent_data in dets_df.groupby('det_id'):
            det_obs_dict[f'{det_agent}_orientation'] = [np.nan]*len(obs_frames)
            det_obs_dict[f'{det_agent}_val'] = [False]*len(obs_frames)
            det_obs_dict[f'{det_agent}_x'] = [np.nan]*len(obs_frames)
            det_obs_dict[f'{det_agent}_y'] = [np.nan]*len(obs_frames)
            det_obs_dict[f'{det_agent}_img_x'] = [np.nan]*len(obs_frames)
            det_obs_dict[f'{det_agent}_img_y'] = [np.nan]*len(obs_frames)
            assert det_agent_data.original_id.nunique() == 1, 'Unexpected number of original id'
            orig_id = det_agent_data.original_id.unique()[0]
            det_obs_dict[f'{det_agent}_orig_id'] = [orig_id]*len(obs_frames)
        det_obs_df = pd.DataFrame(det_obs_dict)
        det_obs_df.index = det_obs_df.frame_id
        det_obs_df = det_obs_df.drop(columns=['frame_id'])
        for det_agent, det_agent_data in dets_df.groupby('det_id'):
            # For detection, use camera relative yaw
            det_obs_df.loc[det_agent_data.frame_id, [f'{det_agent}_x', f'{det_agent}_y', f'{det_agent}_orientation']] = \
                det_agent_data[['x_w', 'z_w', 'yaw_w']].values
            det_img_x = det_agent_data['bbox_left'] + 0.5*det_agent_data['bbox_width']
            det_img_y = det_agent_data['bbox_top'] + 0.5*det_agent_data['bbox_height']
            det_obs_df.loc[det_agent_data.frame_id, f'{det_agent}_img_x'] = det_img_x.values
            det_obs_df.loc[det_agent_data.frame_id, f'{det_agent}_img_y'] = det_img_y.values
            det_obs_df.loc[det_agent_data.frame_id, f'{det_agent}_val'] = True
        det_obs_df = det_obs_df[natsorted(det_obs_df.columns)]

        # Filter out detections which have fewer than "min detection len" occurrences...
        # Interpolate those left that may have at least one allowed NaN
        if det_obs_df.isnull().values.any():
            to_drop = []
            for col in det_obs_df.columns:
                # Precisely 1 non-NaN -> fill with nearest
                num_valid = det_obs_df[col].notnull().sum()
                if num_valid < self.min_detection_len:
                    col_agent = col.split('_')[0]
                    # These columns always have valid values, so need to be dropped 
                    # explicitly if the agent's other cols are missing
                    col_val = f'{col_agent}_val'
                    col_orig = f'{col_agent}_orig_id'
                    to_drop.append(col)
                    if not col_val in to_drop:
                        to_drop.append(col_val)
                    if not col_orig in to_drop:
                        to_drop.append(col_orig)
                    continue
                if num_valid == 1:
                    det_obs_df[col] = det_obs_df[col].interpolate(limit_direction='both')
            det_obs_df = det_obs_df.drop(columns=to_drop)
            # For now, assume okay to interpolate e.g. yaw, bbox centers, etc. since valid marked False
            det_obs_df = det_obs_df.interpolate(method="slinear", fill_value="extrapolate", limit_direction="both")
        det_obs_vals = det_obs_df.values
        # Sorted order will be ['img_x', 'img_y', 'orientation', 'orig_id', 'val', 'x', 'y']
        # Final order will be ['x', 'y', 'orientation', 'img_x', 'img_y', 'val', 'orig_id']
        det_obs_img_x = det_obs_vals[:, ::7].transpose()
        det_obs_img_y = det_obs_vals[:, 1::7].transpose()
        det_obs_orient = det_obs_vals[:, 2::7].transpose()
        det_obs_orig = det_obs_vals[:, 3::7].transpose().astype(int)
        det_obs_val = det_obs_vals[:, 4::7].transpose().astype(int)
        det_obs_x = det_obs_vals[:, 5::7].transpose()
        det_obs_y = det_obs_vals[:, 6::7].transpose()
        # TODO: validate all this interpolation stuff once more pls?? Wow...
        det_obs = np.stack([det_obs_x, det_obs_y, det_obs_orient, 
                            det_obs_img_x, det_obs_img_y, det_obs_val, det_obs_orig], axis=-1)

        # Now handling ground truth
        all_frames = np.concatenate([obs_frames, pred_frames])
        gt_det_pred_dict = {'frame_id': all_frames}
        gt_det_agents = {}
        # Rule: any agent with at least "min detection len" detections in obs phase
        # Really: just trying to get, in the 
        for obs_frame in obs_frames:
            tmp_df = full_dict_gt[agent_id][obs_frame]
            if not len(tmp_df):
                continue
            # col_names = ['det_id', ..., 'num_pixels']
            # care about first column and last column
            for row in tmp_df:
                if row[-1] < self.vis_ratio:
                    continue
                gt_det_agent_id = row[0]
                if gt_det_agent_id in gt_det_agents:
                    gt_det_agents[gt_det_agent_id] += 1
                else:
                    gt_det_agents[gt_det_agent_id] = 1

        gt_det_agents = [k for k, v in gt_det_agents.items() if v >= self.min_detection_len]
        # These agents have at least "min_detection_len" observed points in observation
        for gt_det_agent in gt_det_agents:
            gt_det_pred_dict[f'{gt_det_agent}_orientation'] = [np.nan]*len(all_frames)
            gt_det_pred_dict[f'{gt_det_agent}_val'] = [False]*len(all_frames)
            gt_det_pred_dict[f'{gt_det_agent}_x'] = [np.nan]*len(all_frames)
            gt_det_pred_dict[f'{gt_det_agent}_y'] = [np.nan]*len(all_frames)
            gt_det_pred_dict[f'{gt_det_agent}_img_x'] = [np.nan]*len(all_frames)
            gt_det_pred_dict[f'{gt_det_agent}_img_y'] = [np.nan]*len(all_frames)
            gt_det_pred_dict[f'{gt_det_agent}_orig_id'] = [gt_det_agent]*len(all_frames)
        gt_det_pred_df = pd.DataFrame(gt_det_pred_dict)
        gt_det_pred_df.index = gt_det_pred_df.frame_id
        gt_det_pred_df = gt_det_pred_df.drop(columns=['frame_id'])
        # Actual ground truth locations of where people are (not who can see whom)
        gt_ref = full_gt[(full_gt.agent_id.isin(gt_det_agents)) &\
                        (full_gt.frame_id.isin(all_frames))]
        # Now just want their full information
        for gt_det_agent, gt_det_agent_data in gt_ref.groupby('agent_id'):
            frame_ids = gt_det_agent_data.frame_id.values
            gt_det_pred_df.loc[frame_ids, [f'{gt_det_agent}_x', f'{gt_det_agent}_y', f'{gt_det_agent}_orientation']] = \
                gt_det_agent_data[['x', 'z', 'yaw']].values
            # Unknown relative to `agent`, so placing at center for now
            gt_det_pred_df.loc[frame_ids, f'{gt_det_agent}_img_x'] = self.img_width/2
            gt_det_pred_df.loc[frame_ids, f'{gt_det_agent}_img_y'] = self.img_height/2
            # There are MORE valid in gt than in hist/fut BECAUSE even though the vis_ratio check ensures alignment,
            # the ACTUAL BEV ground truth locations for those situations where the person is NOT visible are used!!!!!
            gt_det_pred_df.loc[frame_ids, f'{gt_det_agent}_val'] = True
        gt_det_pred_df = gt_det_pred_df[natsorted(gt_det_pred_df.columns)]
        gt_det_pred_vals = gt_det_pred_df.values
        gt_det_pred_img_x = gt_det_pred_vals[:, ::7].transpose()
        gt_det_pred_img_y = gt_det_pred_vals[:, 1::7].transpose()
        gt_det_pred_orient = gt_det_pred_vals[:, 2::7].transpose()
        gt_det_pred_orig_id = gt_det_pred_vals[:, 3::7].transpose()
        gt_det_pred_val = gt_det_pred_vals[:, 4::7].transpose().astype(int)
        gt_det_pred_x = gt_det_pred_vals[:, 5::7].transpose()
        gt_det_pred_y = gt_det_pred_vals[:, 6::7].transpose()
        gt_det_pred = np.stack([gt_det_pred_x, gt_det_pred_y, gt_det_pred_orient, 
                                gt_det_pred_img_x, gt_det_pred_img_y, gt_det_pred_val, gt_det_pred_orig_id], axis=-1)
        det_pred = gt_det_pred

        if self.hungarian_match:
            # Cost matrix will be (x, y) ADE over observation period
            # Filter out nan from det_pred's future period too
            det_pred_filt = [x for x in det_pred if not np.isnan(x[:, :2].astype(np.float32)).any()]
            if not len(det_pred_filt) or not len(det_obs):
                return np.empty((0, *det_obs.shape[1:])), np.empty((0, *det_pred.shape[1:])), []
            det_pred = np.stack(det_pred_filt)
            traj_pred = det_pred[:, :self.hist_len, :2].astype(np.float32)
            traj_obs = det_obs[:, :self.hist_len, :2].astype(np.float32)
            # n_pred x n_obs
            cost_ade = np.sum(np.sqrt(np.sum((traj_obs - traj_pred[:, np.newaxis, :, :]) **2, axis=-1)), axis=-1) / self.hist_len
            row_ind, col_ind = linear_sum_assignment(cost_ade)
            
            new_det_obs = []
            new_det_pred = []
            gt_ids = []
            for new_agent_id, (pred_idx, obs_idx) in enumerate(zip(row_ind, col_ind)):
                new_det_obs.append(det_obs[obs_idx])
                new_det_pred.append(det_pred[pred_idx])
                new_det_obs[-1][..., -1] = new_agent_id
                new_det_pred[-1][..., -1] = new_agent_id
                gt_ids.append(new_agent_id)
            det_obs = np.stack(new_det_obs)
            det_pred = np.stack(new_det_pred)
        
        if not self.hungarian_match and self.alignment and self.noise is not None:
            new_det_pred = np.zeros((len(det_obs), *(det_pred.shape[1:])))
            gt_ids = []
            for idx in range(len(det_obs)):
                id_to_find = det_obs[idx, 0, -1]
                gt_to_search = det_pred[:, 0, -1]
                gt_idx = np.nonzero(gt_to_search == id_to_find)[0]
                assert gt_idx.shape == (1,), 'Unexpected number of matches for shape'
                gt_idx = gt_idx.item()
                new_det_pred[idx] = det_pred[gt_idx]
                # OVerwrite orig_id for alignment
                det_obs[idx, :, -1] = idx
                new_det_pred[idx, :, -1] = idx
                gt_ids.append(idx)
            return det_obs, new_det_pred, gt_ids

        if not self.hungarian_match:
            gt_ids = natsorted(list(set([int(x.split('_')[0]) for x in gt_det_pred_df.columns])))
            if self.alignment:
                det_ids = natsorted(list(set([int(x.split('_')[0]) for x in det_obs_df.columns])))
                assert det_ids == gt_ids, 'Mismatch in alignment between GT and Det assumption; confirm config'
        return det_obs, det_pred, gt_ids

FPVDataset.cached_input_data = None

    
# TODO: use scenes vs agent? answer! Just incorporate "scene" analysis at end of each epoch!
def fpv_seq_collate(data):
    (
        obs_list,
        pred_list,
        obs_resnet_list,
    ) = zip(*data)
    def get_start_end(input_list):
        _len = [seq.shape[0] for seq in input_list]
        cum_start_idx = [0] + np.cumsum(_len).tolist()
        start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        return torch.from_numpy(np.array(start_end))
    obs_seq_start_end = get_start_end(obs_list)
    pred_seq_start_end = get_start_end(pred_list)

    # shapes of each element of obs_list: N_det x 8 x 2, pred_list: N_pred x 12 x 2
    # Desired LSTM shape: seq_len x batch x input_size
    obs_seq = torch.cat(obs_list).permute(1, 0, 2)
    pred_seq = torch.cat(pred_list).permute(1, 0, 2)

    #obs_resnet = torch.cat(obs_resnet_list).permute(1, 0, 2)
    obs_resnet = torch.cat([torch.stack([x_res for _ in range(len(x_obs))]) \
                               for x_obs, x_res in zip(obs_list, obs_resnet_list)]).permute(1, 0, 2)
    return tuple([obs_seq, obs_resnet, obs_seq_start_end, pred_seq, pred_seq_start_end])

def fpv_dataset_name(data_config, traj_config, out_name):
    if 'balance' in data_config and data_config['balance']['enabled']:
        out_name = out_name.replace('.npy', f'_bal.npy')
    if 'min_detection_len' in data_config:
        out_name = out_name.replace('.npy', f'_mdl-{data_config.min_detection_len}.npy')
    if data_config.load_images:
        out_name = out_name.replace('.npy', f'_imgs.npy')
    if data_config.load_detections:
        out_name = out_name.replace('.npy', f'_dets.npy')
    if data_config.alignment:
        out_name = out_name.replace('.npy', f'_align.npy')
    if data_config.load_resnet:
        out_name = out_name.replace('.npy', f'_resnet.npy')
    if 'hungarian_match' in data_config and data_config['hungarian_match']:
        out_name = out_name.replace('.npy', f'_hmatch.npy')

    if 'pred_dets' in data_config and data_config['pred_dets']:
        out_name = out_name.replace('.npy', '_pred-dets.npy')
    
    if 'noise' in data_config and data_config['noise']['enabled']:
        noise_config = mutils.dotdict(data_config['noise'])
        noise_str = f'mu-{noise_config.mu}_std-{noise_config.std}_df-{noise_config.drop_frame}_'\
                    f'dt-{noise_config.drop_traj}_lt-{noise_config.lose_traj}_sd-{noise_config.seed}'
        out_name = out_name.replace('.npy', f'_{noise_str}.npy')
    return out_name

