# ------------------------------------------------------------------------------
# @file:    base_trainer.py
# @brief:   This file contains the implementation of the BaseTrainer class which
#           is used as a base class for implementing Trajectory Prediction 
#           trainers.  
# @author:  Ingrid Navarro, Ben Stoler
# @date:    Last modified on August 3rd, 2022
# ------------------------------------------------------------------------------

import json
import logging
import os
import math
import numpy as np
import torch

from natsort import natsorted

from torch.utils.tensorboard import SummaryWriter

# sprnn modules
import vrnntools.utils.common as mutils    

from vrnntools.utils import visualization as vis
from vrnntools.utils.common import Config, DIMS
from vrnntools.utils.data_loader import load_data

FORMAT = '[%(asctime)s: %(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logger = logging.getLogger(__name__)

seed = 1
np.random.seed(seed)

class BaseTrainer:
    """ A class that implements base trainer methods. """
    def __init__(self, config: dict) -> None:
        """ Initializes the trainer.
        Inputs:
        -------
        config[dict]: a dictionary containing all configuration parameters.
        """
        self._config = Config(config)
        super().__init__()
        
        # set random seed
        seed = self.config.TRAIN.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.trainer = self.config.BASE_CONFIG.trainer
        self.coord = self.config.BASE_CONFIG.coord
        self.dim = self.config.MODEL.dim
        self.retrack = self.config.BASE_CONFIG.retrack
        assert self.dim in DIMS, f"Dimension: {self.dim} not supported ({DIMS})"

        self.setup_outputs()
        
        # saves a copy of the config file used for the current experiment
        json_filename = os.path.join(self.out.base, 'config.json')
        with open(json_filename, 'w') as json_file:
            json.dump(config, json_file, indent=2)
        logger.info(f"Config:\n{json.dumps(config, indent=2)}")
        logger.info(f"{self.name} saved the configuration to: {json_filename}")
        
        # loads or generates dataset using config specifications
        self.multi_test = self.config.BASE_CONFIG.multi_test and (self.config.BASE_CONFIG.run_type != 'data')
        if not self.multi_test:
            self.train_data, self.val_data, self.test_data = load_data(
                self.config.DATASET, self.config.TRAJECTORY)
        else:
            self.train_data, self.val_data, _ = load_data(
                self.config.DATASET, self.config.TRAJECTORY)
            self.test_data = None
            assert len(self.config.BASE_CONFIG.test_info), 'Must include at least one test'
            self.all_test_data = {}
            for test_info in self.config.BASE_CONFIG.test_info:
                assert os.path.exists(test_info['path']), f'Test config missing: {test_info["path"]}'
                # Overwrite e.g. eth -> hotel, etc.
                fold_name = self.config.BASE_CONFIG['dataset']['name']
                with open(test_info['path']) as f:
                    test_config = json.load(f)
                    test_config['dataset']['name'] = fold_name
                    if self.retrack:
                        test_config['dataset']['alignment'] = False
                        test_config['dataset']['hungarian_match'] = False
                single_test_data = load_data(mutils.dotdict(test_config['dataset']),
                                             mutils.dotdict(test_config['trajectory']),
                                             test_only=True)
                self.all_test_data[test_info['name']] = single_test_data
                if self.test_data is None:
                    self.test_data = single_test_data
                    self.test_num_iter = len(self.test_data)
        if hasattr(self.train_data, '__len__') and len(self.train_data):
            train_len, val_len = len(self.train_data), len(self.val_data)
            test_len = len(self.test_data)
            logger.info(
                f"Dataset size - train: {train_len} val: {val_len} test: {test_len}")
        else:
            train_len, val_len, test_len = np.inf, np.inf, np.inf
        self.train_batch_size = self.config.DATASET.train_batch_size
        self.val_batch_size = self.config.DATASET.val_batch_size
        self.test_batch_size = self.config.DATASET.test_batch_size
        
        self.device = (
            torch.device("cuda", self.config.BASE_CONFIG.gpu_id)
            if torch.cuda.is_available() and not self.config.BASE_CONFIG.use_cpu
            else torch.device("cpu")
        )
        logger.info(f"{self.name} uses torch.device({self.device})")

        # training parameters
        self.batch_size = self.config.TRAIN.batch_size
        #self.val_batch_size = self.config.TRAIN.val_batch_size
        self.num_samples = self.config.TRAIN.num_samples
        self.scene_metrics = 'scene_metrics' in self.config.TRAIN and self.config.TRAIN.scene_metrics

        if self.config.TRAIN.num_iter < 0:
            self.config.TRAIN.num_iter = train_len
        self.num_iter = min(self.config.TRAIN.num_iter, train_len)
        
        if self.config.TRAIN.eval_num_iter < 0:
            self.config.TRAIN.eval_num_iter = val_len
        self.eval_num_iter = min(self.config.TRAIN.eval_num_iter, val_len)
        
        if self.config.TRAIN.test_num_iter < 0:
            self.config.TRAIN.test_num_iter = test_len
        self.test_num_iter = min(self.config.TRAIN.test_num_iter, test_len)

        # Priority: train < model override < command line override
        self.num_epoch = self.config.TRAIN.num_epoch
        if 'overwrite_epochs' in self.config.MODEL:
            self.num_epoch = self.config.MODEL.overwrite_epochs
        if 'n_epoch' in self.config.BASE_CONFIG and self.config.BASE_CONFIG.n_epoch > 0:
            self.num_epoch = self.config.BASE_CONFIG.n_epoch
        
        self.hist_len = self.config.TRAJECTORY.hist_len
        self.fut_len = self.config.TRAJECTORY.fut_len
        
        self.traj_len = self.hist_len + self.fut_len
        self.pat_len = 10
        if not self.config.TRAJECTORY.pat_len == None:
            self.pat_len = self.config.TRAJECTORY.pat_len
        
        # specifies the number of warm-up epochs to use during training, here
        # the KLD loss will be reduced during warm-up
        self.warmup = np.ones(self.num_epoch)
        if self.config.TRAIN.warmup:
            warmup_epochs = self.config.TRAIN.warmup_epochs
            self.warmup_epochs = warmup_epochs
            self.warmup[:warmup_epochs] = np.linspace(0, 1, num=warmup_epochs)
        
        self.ade_loss = 'ade_loss' in self.config.TRAIN and self.config.TRAIN.ade_loss
        if self.ade_loss:
            import pdb; pdb.set_trace()

        self.gradient_clip = (
            self.config.TRAIN.gradient_clip if self.config.TRAIN.gradient_clip else 10)

        self.patience = (
            self.config.TRAIN.patience if self.config.TRAIN.patience else 10)
        
        self.update_lr = self.config.TRAIN.update_lr
        self.dataset_name = self.config.DATASET.name
        
        self.visualize = self.config.VISUALIZATION.enabled
        self.plot_freq = self.config.VISUALIZATION.plot_freq
        
        # measures and losses
        self.collision_radius = self.config.TRAJECTORY.collision_radius
        self.interp_valid_only = 'interp_valid_only' in self.config.MODEL and self.config.MODEL.interp_valid_only
        self.train_gt = self.config.MODEL.train_gt if 'train_gt' in self.config.MODEL else False
        self.train_e2e = self.config.MODEL.train_e2e if 'train_e2e' in self.config.MODEL else False
        self.train_corr = self.config.BASE_CONFIG.train_corr if 'train_corr' in self.config.BASE_CONFIG else False
        self.smoothing = self.config.TRAJECTORY.smoothing if 'smoothing' in self.config.TRAJECTORY else False

    @property
    def config(self) -> Config:
        return self._config

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def train(self, do_eval: bool = False, train_corr=False) -> None:
        """ Base training process. The method specific to the trainer is 
        train_epoch().
        Inputs:
        -------
        do_eval[bool]: if True it will run eval_epoch() after train_epoch().
        """
        logger.info("{} training details:\n{}".format(self.name, json.dumps(
            self.config.TRAIN, indent=2)))

        best_error = float('inf')
        start_epoch = 0
        best_epoch = -1
        
        # check if start training from a previous checkpoint
        if self.config.TRAIN.load_model:
            ckpt_file = os.path.join(self.out.ckpts, self.config.TRAIN.ckpt_name)
            assert os.path.exists(ckpt_file), \
                f"Checkpoint {ckpt_file} does not exist!"

            self.load_model(ckpt_file)

            start_epoch = int(self.config.TRAIN.ckpt_name.split('_')[-1].split('.')[0])

        # start training
        for epoch in range(start_epoch, self.num_epoch):
        
            epoch_str = f"Epoch[{epoch+1}/{self.num_epoch}]\n\ttrain: "
            loss = self.train_epoch(epoch, train_corr=train_corr)
            assert not math.isnan(loss['Loss']), f"Nan-loss at epoch: {epoch}"
        
            # write loss to tensorboard
            for k, v in loss.items():
                epoch_str += f"{k}: {round(v, 5)} "
                self.tb_writer.add_scalar(f'Train/{k}', v, epoch)
        
            save_best_ckpt = False
            #if do_eval and (epoch % self.config.TRAIN.ckpt_freq == 0):
            if do_eval:
                epoch_str += "\n\teval: "
                self.model.eval()
                measures = self.eval_epoch(epoch, num_samples=self.num_samples, train_corr=train_corr)
                
                if hasattr(self, 'eval_metrics') and hasattr(self.eval_metrics, 'main_metric'):
                    main_metric = self.eval_metrics.main_metric
                else:
                    main_metric = 'MinADE'

                if self.update_lr:
                    self.lr_scheduler.step(measures[main_metric])
                    
                for k, v in measures.items():
                    assert not math.isnan(v), f"{k} got nan at epoch: {epoch}"
                    epoch_str += f"{k}: {round(v, 5)} "
                    self.tb_writer.add_scalar(f'Val/{k}', v, epoch)
                    
                if measures[main_metric] < best_error:
                    save_best_ckpt = True
                    best_error = measures[main_metric]
                    epoch_str += " new best"
                    best_epoch = epoch
                    

            # save current model to checkpoint
            if (epoch+1) % self.config.TRAIN.ckpt_freq == 0 or save_best_ckpt:
                self.save_model(epoch+1)
                
            logger.info(f"{epoch_str}")
        epoch_str = f"Epoch[{best_epoch+1}/{self.num_epoch}]"
        if train_corr:
            best_file = os.path.join(self.out.ckpts, f'ckpt_{best_epoch+1}_corr.pth')
        else:
            best_file = os.path.join(self.out.ckpts, f'ckpt_{best_epoch+1}.pth')
        self.load_model(best_file)
        def test_one(key=None):
            nonlocal epoch_str
            test_measures = self.test_epoch(epoch, num_samples=self.num_samples, train_corr=train_corr)
            if key is None:
                epoch_str += "\n\ttest: "
            else:
                epoch_str += f"\n\ttest ({key}): "
            for k, v in test_measures.items():
                assert not math.isnan(v), f"{k} got nan at epoch: {epoch}"
                epoch_str += f"{k}: {round(v, 5)} "
                if key is None:
                    self.tb_writer.add_scalar(f'Test/{k}', v, epoch)
                else:
                    self.tb_writer.add_scalar(f'Test-{key}/{k}', v, epoch)
        if not self.multi_test:
            test_one()
        else:
            test_names = natsorted([x for x in self.all_test_data.keys()])
            for test_name in test_names:
                test_loader = self.all_test_data[test_name]
                self.test_data = test_loader
                self.test_num_iter = len(test_loader)
                test_one(test_name)
        logger.info(f"{epoch_str}")
        # Run test once, on the best epoch!

    def eval(self, do_eval=True, train_corr=False, retrack=False, load_only=False) -> None:
        """ Evaluates all checkpoints from the corresponding experiment. The 
        method specific to the trainer is eval_epoch(). 
        Inputs:
        -------
        do_eval[boolean]: if True it will run validation, otherwise it will 
        run testing.
        """
        tb_name = 'Val'if do_eval else 'Test'
        data_split = 'val' if do_eval else 'test'
        logger.info(f"Running evaluation on {tb_name}!")
        
        best_line = ''
        # TODO: break up this logic into helper functions?
        if self.config.BASE_CONFIG.load_ckpt:
            if self.config.BASE_CONFIG.ckpt_name:
                ckpt_files = [self.config.BASE_CONFIG.ckpt_name]
            else:
                # Load the best checkpoint from the most recent training
                if train_corr:
                    trainval_path = self.out['ckpts'].replace('/ckpts', '/traincorr.log')
                else:
                    trainval_path = self.out['ckpts'].replace('/ckpts', '/trainval.log')
                assert os.path.exists(trainval_path), 'Could not find previous training instance'
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
                    if ade < best_ade and cur_epoch <= self.config.BASE_CONFIG['max_test_epoch']:
                        best_epoch = cur_epoch
                        best_ade = ade
                        best_line = line
                assert best_epoch > 0, 'Could not find best epoch to load from'
                
                # 1-indexed already
                if train_corr:
                    ckpt_files = [f'ckpt_{best_epoch}_corr.pth']
                else:
                    ckpt_files = [f'ckpt_{best_epoch}.pth']
        else:
            ckpt_files = natsorted(os.listdir(self.out.ckpts))
            logging.info(f"Running checkpoints from dir: {self.out.ckpts}")
            
        assert len(ckpt_files) > 0, f"No checkpoints in dir: {self.out.ckpts}"
        # Only eval on best one
        ckpt_files = [ckpt_files[-1]]
        
        for file in ckpt_files:
            ckpt_file = os.path.join(self.out.ckpts, file)
            self.load_model(ckpt_file)
            if load_only:
                return ckpt_file
            
            logger.info(f"{self.name} running checkpoint: {file}")
            epoch = int(file.split('ckpt_')[-1].split('.')[0].split('_')[0])
            epoch_str = f"Epoch[{epoch+1}/{self.num_epoch}] "
            if len(best_line):
                epoch_str += f'\n{best_line.rstrip()}'
            
            import shutil        
            shutil.rmtree(self.out['trajs'])
            os.makedirs(self.out['trajs'])
            def test_one(key=None):
                nonlocal epoch_str
                print(key)
                test_measures = self.test_epoch(epoch, num_samples=self.num_samples, train_corr=train_corr)
                if key is None:
                    epoch_str += "\n\ttest: "
                else:
                    epoch_str += f"\n\ttest ({key}): "
                for k, v in test_measures.items():
                    assert not math.isnan(v), f"{k} got nan at epoch: {epoch}"
                    epoch_str += f"{k}: {round(v, 5)} "
                    if key is None:
                        self.tb_writer.add_scalar(f'Test/{k}', v, epoch)
                    else:
                        self.tb_writer.add_scalar(f'Test-{key}/{k}', v, epoch)
            if not self.multi_test:
                test_one()
            else:
                test_info = self.all_test_data
                for name, loader in test_info.items():
                    n_ego = loader.dataset.obs_seq_start_end.shape[0]
                    n_obs = loader.dataset.all_obs.shape[0] - n_ego
                    n_pred = loader.dataset.all_pred.shape[0] - n_ego
                    print(f'{name}: {n_ego} ego, {n_obs}/{n_pred} dets ({n_obs/n_pred:.2f})')
                test_names = natsorted([x for x in self.all_test_data.keys()])
                for test_name in test_names:
                    test_loader = self.all_test_data[test_name]
                    self.test_data = test_loader
                    self.test_num_iter = len(test_loader)
                    test_one(test_name)
            logger.info(f"{epoch_str}")

    def setup_outputs(self) -> None:
        """ Creates the experiment name-tag and all output directories. """
        # create the experiment tag name

        # TODO: adjust this appropriately, decide on conventions
        exp_name = "{}_{}_{}_{}d_hl-{}_hs-{}_fl-{}_fs-{}".format(
            self.config.BASE_CONFIG.exp_tag, 
            self.trainer, 
            self.coord, 
            self.dim, 
            self.config.TRAJECTORY.hist_len, 
            self.config.TRAJECTORY.hist_step,  
            self.config.TRAJECTORY.fut_len, 
            self.config.TRAJECTORY.fut_step
        )

        # create all output directories
        out = os.path.join(
            self.config.BASE_CONFIG.out_dir, self.config.DATASET.name, exp_name)
        if not os.path.exists(out):
            os.makedirs(out)

        # create subdirs required for the experiments
        assert not self.config.BASE_CONFIG.sub_dirs == None, \
            f"No sub-dirs were specified!"

        self.out = {}
        for sub_dir in self.config.BASE_CONFIG.sub_dirs:
            self.out[sub_dir] = os.path.join(out, sub_dir)
            if not os.path.exists(self.out[sub_dir]):
                os.makedirs(self.out[sub_dir])

        # TODO: move this to another place...like start of test_epoch
        # Clean up old saved trajectories...
        # import shutil        
        # shutil.rmtree(self.out['trajs'])
        # os.makedirs(self.out['trajs'])

        self.out = mutils.dotdict(self.out)
        self.out.base = out
        self.config.VISUALIZATION.plot_path = self.out.plots
        self.config.VISUALIZATION.video_path = self.out.videos

        # tensorboard writer
        tb_dir = os.path.join(out, 'tb')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        self.tb_writer = SummaryWriter(tb_dir)
        
        level = (logging.DEBUG 
            if self.config.BASE_CONFIG.log_mode == "debug" else logging.INFO)
        output_log = os.path.join(out, self.config.BASE_CONFIG.log_file)
        logging.basicConfig(
            filename=output_log, filemode='a', level=level, format=FORMAT, 
            datefmt='%Y-%m-%d %H:%M:%S')
        
        logger.info(f"{self.name} created output directory: {out}")
    
    def save_model(self, epoch: int) -> None:
        """ Saves a predictor model, optimizer and lr scheduler to specified 
        filename.
        Inputs:
        -------
        epoch[int]: epoch number of corresponding model.
        """
        if self.train_corr:
            ckpt_file = os.path.join(self.out.ckpts, f'ckpt_{epoch}_corr.pth')
        else:
            ckpt_file = os.path.join(self.out.ckpts, f'ckpt_{epoch}.pth')
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()}, ckpt_file)
        logger.info(f"{self.name} saved checkpoint to: {ckpt_file}")
        self.save_impl(epoch)
       
    def load_model(self, filename: str) -> None:
        """ Loads a predictor model from specified filename.
        Inputs:
        -------
        filename[str]: checkpoint filename. 
        """
        logger.debug(f"{self.name} loading checkpoint from: {filename}")
        model = torch.load(filename, map_location=torch.device('cpu'))
        self.model.load_state_dict(model['model'])
        self.optimizer.load_state_dict(model['optimizer'])
        self.lr_scheduler.load_state_dict(model['lr_scheduler'])
    
    def save_tensors(self, tensors_out: dict, batch_idx: int, epoch: int, label: str):
        """ Saves specified tensors into npy arrays for offline analyses
        """
        for k, v in tensors_out.items():
            tensors_out[k] = v.detach().cpu().numpy()
        file_name = f'{label}_epoch{epoch}_batch{batch_idx}.npy'
        path_out = os.path.join(self.out.trajs, file_name)
        np.save(path_out, tensors_out)
    
    def generate_outputs(
        self, hist: torch.tensor, fut: torch.tensor, preds: torch.tensor, 
        best_sample_idx: torch.tensor, seq_start_end: torch.tensor, 
        filename: str, epoch: int): 
        """ Generates visualizations for ground-truth and predicted trajectories.
        And saves predictions into npy arrays. 
        
        Inputs:
        -------
        hist[torch.tensor]: trajectory observed history
        fut[torch.tensor]: trajectory future
        preds[torch.tensor]: predicted trajectories
        best_sample_idx[torch.tensor]: index of best predicted trajectory based
            on MinADE. 
        seq_start_end[torch.tensor]: tensor indicating where scenes start/end.
        filename[str]: plot filename. 
        """
        # TODO: update to work with new dataset and such
        import pdb; pdb.set_trace()
        if not self.visualize:
            return

        # pred = pred_list[best_sample_idx]
        preds = preds.cpu() if preds.is_cuda else preds
        preds = torch.transpose(preds, 2, 3).numpy()
        # np.save(
        #     os.path.join(self.out.trajs, f"traj-{i}_pred.npy"), pred)
    
        hist = hist.cpu() if hist.is_cuda else hist
        hist = torch.transpose(hist, 1, 2).numpy()
        # np.save(
        #     os.path.join(self.out.trajs, f"traj-{i}_hist.npy"), hist)
        
        fut = fut.cpu() if fut.is_cuda else fut
        fut = torch.transpose(fut, 1, 2).numpy()
        # np.save(
        #     os.path.join(self.out.trajs, f"traj-{i}_fut.npy"), fut)
        
        best_sample_idx = (best_sample_idx.cpu() 
            if best_sample_idx.is_cuda else best_sample_idx)
        
        # vis.plot_trajectories(
        #     self.config.VISUALIZATION, hist, fut, preds, seq_start_end, 
        #     best_sample_idx.numpy(), filename)
    
    # --------------------------------------------------------------------------
    # All methods below should be implemented by each trainer that inherits from 
    # the BaseTrainer class.
    # --------------------------------------------------------------------------
    def save_impl(self, epoch: int):
        pass

    def train_epoch(self, epoch: int, **kwargs) -> dict:
        """ Trains one epoch. 
        Inputs:
        -------
        epoch[int]: epoch number to test. 
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all of losses computed during training. 
        """
        error_msg = f"train_epoch() should be implemented by {self.name}"
        raise NotImplementedError(error_msg)

    def eval_epoch(self, epoch: int, **kwargs) -> dict:
        """ Evaluates one epoch. 
        Inputs:
        -------
        epoch[int]: epoch number to test. 
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all of losses computed during training. 
        """
        error_msg = f"eval_epoch() should be implemented by {self.name}"
        raise NotImplementedError(error_msg)

    def test_epoch(self, epoch: int, **kwargs) -> dict:
        """ Evaluates one epoch. 
        Inputs:
        -------
        epoch[int]: epoch number to test. 
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all of losses computed during training. 
        """
        error_msg = f"test_epoch() should be implemented by {self.name}"
        raise NotImplementedError(error_msg)

    def compute_loss(self, **kwargs) -> float:
        """ Computes trainer loss. 
        Inputs:
        -----------
        **kwargs: Keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[float]: epoch's total loss. 
        """
        error_msg = f"compute_loss() should be implemented by {self.name}"
        raise NotImplementedError(error_msg)

    def setup(self) -> None:
        """ Initializes the model, optimizer, lr_scheduler, etc. """
        error_msg = f"setup() should be implemented by {self.name}"
        raise NotImplementedError(error_msg)
