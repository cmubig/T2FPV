# ------------------------------------------------------------------------------
# @file:    data_loader.py
# @brief:   Contains utility functions to preprocess the datasets.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# general includes
import logging
import os
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import vrnntools.utils.common as mutils

from vrnntools.utils.datasets.fpv_dataset import FPVDataset, fpv_seq_collate, fpv_dataset_name
from vrnntools.utils.datasets.bev_dataset import BEVDataset, bev_seq_collate, bev_dataset_name
# ------------------------------------------------------------------------------
# local includes

# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

def load_data(data_config: dict, traj_config: dict, test_only=False):
    """ Loads the data for training, validating and testing depending on the
    configuration. 
    Inputs:
        config: Contains all configuration parameters needed to load the data
    Outputs:
        data loaders for training, validation and testing.
    """
    npy_path = data_config.npy_path
    name = data_config.name
    loader_type = data_config.loader_type

    # Name-tag the cache file
    hl, fl = traj_config.hist_len, traj_config.fut_len
    hs, fs = traj_config.hist_step, traj_config.fut_step
    # Min agents with greater than or equal to
    min_agents, sk = traj_config.min_agents, traj_config.skip
    out_name = f"{name}_{loader_type}_hl-{hl}_fl-{fl}_hs-{hs}_fs-{fs}_sk-{sk}_ag-{min_agents}.npy"
    
    if loader_type == 'fpv':
        out_name = fpv_dataset_name(data_config, traj_config, out_name)
    elif loader_type == 'bev':
        out_name = bev_dataset_name(data_config, traj_config, out_name)
    else:
        raise NotImplementedError(f"Loader type {loader_type} not implemented")

    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    npy_file = os.path.join(npy_path, out_name)
    
    if data_config.load_npy:
        if os.path.exists(npy_file):
            logger.info(f"Loading data from {npy_file}...")
            train_loader, val_loader, test_loader = np.load(npy_file, allow_pickle=True)
            num_workers = data_config.loader_num_workers
            def update_loader(loader, batch_size):
                shuffle = isinstance(loader.sampler, RandomSampler)
                return DataLoader(loader.dataset, batch_size=batch_size, num_workers=num_workers,
                                    shuffle=shuffle, collate_fn=loader.collate_fn)
            #rand_el = train_loader.dataset[int(np.random.rand()*len(train_loader.dataset))]
            if not test_only:
                return update_loader(train_loader, data_config.train_batch_size),\
                        update_loader(val_loader, data_config.val_batch_size), \
                        update_loader(test_loader, data_config.test_batch_size)
            else:
                return update_loader(test_loader, data_config.test_batch_size)
        logger.info(f"{npy_file} does not exist. Preprocessing data instead.")

    # Prepare datasets
    if loader_type == 'fpv' or loader_type == 'bev':
        DatasetClass = FPVDataset if loader_type == 'fpv' else BEVDataset
        collate_fn = fpv_seq_collate if loader_type == 'fpv' else bev_seq_collate
        labels = ['train', 'val', 'test']
        label_file_ends = ['_train.txt', '_val.txt', '.txt']
        shuffle_infos = [True, True, False]
        batch_sizes = [data_config.train_batch_size, data_config.val_batch_size, data_config.test_batch_size]

        loaders = []

        # Load trainval by default
        no_trainval = 'no_trainval' in data_config and data_config['no_trainval']
        if no_trainval:
            labels = labels[-1:]
            label_file_ends = label_file_ends[-1:]
            shuffle_infos = shuffle_infos[-1:]
            batch_sizes = batch_sizes[-1:]
            loaders = [None, None]

        for label, label_file_end, shuffle_info, batch_size in zip(labels, label_file_ends, shuffle_infos, batch_sizes):
            logger.info(f"Processing {label} data...")
            dataset_ = DatasetClass(label, label_file_end, traj_config, data_config)
            num_ego = dataset_.seq_len_obs.shape[0]
            num_det = dataset_.all_obs.shape[0] -  num_ego
            num_pred = dataset_.all_pred.shape[0] - num_ego
            print(num_ego, num_det, num_pred, num_det/num_pred)
            logger.info(f"...processed!")
            #rand_el = dataset_[int(np.random.rand()*len(dataset_))]
            loader_ = DataLoader(dataset_,
                                 batch_size=batch_size,
                                 num_workers=data_config.loader_num_workers,
                                 shuffle=shuffle_info,
                                 collate_fn=collate_fn)
            loaders.append(loader_)
        train_loader, val_loader, test_loader = tuple(loaders)
    else:
        raise NotImplementedError(f"Loader type {loader_type} not implemented")

    logger.info(f"Saving data to {npy_file}")
    np.save(npy_file, [train_loader, val_loader, test_loader])
    logger.info(f"Done!")

    if not test_only:
        return train_loader, val_loader, test_loader
    else:
        return test_loader