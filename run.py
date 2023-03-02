import argparse
import json
import os
import subprocess
from copy import deepcopy

from vrnntools.trajpred_trainers.vrnn import VRNNTrainer
from vrnntools.trajpred_trainers.module import ModuleTrainer
from vrnntools.trajpred_trainers.ego_vrnn import EgoVRNNTrainer
from vrnntools.trajpred_trainers.ego_avrnn import EgoAVRNNTrainer
from vrnntools.trajpred_trainers.sgnet_cvae import SGNetCVAETrainer

def get_exp_config(exp_config: str, run_type: str, ckpt: int, fold, gpu_id, use_cpu, max_test_epoch, corr, epochs, no_tqdm):
    # load the configuration files
    assert os.path.exists(exp_config), f"File {exp_config} does not exist!"
    exp_config_file = open(exp_config)
    exp_config = json.load(exp_config_file)

    config_file = open(exp_config["base_config"])
    config = json.load(config_file)
    config.update(exp_config)
    config['max_test_epoch'] = max_test_epoch
    config['n_epoch'] = epochs
    # Keys to overwrite as needed:
    # 1. dataset.name: fold
    # 2. exp_tag: replace 'eth' with fold
    # 3. gpu_id: 
    # 4. use_cpu
    config['dataset']['name'] = fold
    config['exp_tag'] = config['exp_tag'].replace('eth', fold)
    if gpu_id is not None:
        config['gpu_id'] = gpu_id
    if use_cpu:
        config['use_cpu'] = use_cpu

    if not hasattr(config['model_design'], 'items'):
        assert os.path.exists(config['model_design']), 'Model path does not exist'
        with open(config['model_design'], 'r') as f:
            model_config = json.load(f)
        config['model_design'] = model_config
    return config
    

def run_task(params) -> None:
    config = params['config']
    run_type = params['run_type']
    if params['corr']:
        if run_type == 'trainval':
            run_type = 'traincorr'
        elif run_type == 'test':
            run_type = 'testcorr'
        else:
            raise NotImplementedError(f"Run type {run_type} not supported in corr mode")
    ckpt = params['ckpt']
    trainer_type = config['trainer']

    config["log_file"] = f"{run_type}.log"
    config['run_type'] = run_type
    if ckpt:
        config['load_ckpt'] = True
        config['ckpt_name'] = f"ckpt_{ckpt}.pth"
    if (config['run_type'] == 'test' or config['run_type'] == 'testcorr') and not ckpt:
        # Load the best checkpoint from the most recent training
        config['load_ckpt'] = True
        config['ckpt_name'] = False

    config['train_corr'] = run_type in ['traincorr', 'testcorr']
    config['retrack'] = (run_type == 'retrack')
    
    if trainer_type == "vrnn":
        trainer = VRNNTrainer(config=config)
    elif trainer_type == "ego_vrnn":
        trainer = EgoVRNNTrainer(config=config)
    elif trainer_type == "ego_avrnn":
        trainer = EgoAVRNNTrainer(config=config)
    elif trainer_type == "module":
        trainer = ModuleTrainer(config=config)
    elif trainer_type == "sgnet":
        trainer = SGNetCVAETrainer(config=config)
    else:
        raise NotImplementedError(f"Trainer {trainer_type} not supported!")

    
    if run_type == "data":
        import sys; sys.exit(0)

    if run_type == "trainval":
        trainer.train(do_eval=True)
    elif run_type == "traincorr":
        trainer.train(do_eval=True, train_corr=True)
    elif run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "test":
        trainer.eval(do_eval=False)
    elif run_type == "testcorr":
        trainer.eval(do_eval=False, train_corr=True)
    elif run_type == "retrack":
        trainer.eval(do_eval=False, retrack=True)
    else:
        raise NotImplemented(f"Run type {run_type} not supported!")
        
    config['load_ckpt'] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-config', default='./config/fpv_scene/vrnn.json', type=str,
        help='path to experiment configuration file')
    parser.add_argument(
        '--run-type', default='trainval', type=str, choices=['trainval', 'train', 
        'eval', 'test', 'debug', 'data', 'retrack'], help='type of experiment')
    parser.add_argument(
        '--ckpt', required=False, type=int, help='checkpoint number to evaluate')
    parser.add_argument(
        '--fold', default='eth', required=False, help='Which fold to use; overrides config info'
    )
    parser.add_argument(
        '--gpu-id', type=int, required=False, help='Which GPU to use; overrides config info'
    )
    parser.add_argument(
        '--use-cpu', action='store_true', help='Whether to use cpu; overrides config info'
    )
    parser.add_argument(
        '--max-test-epoch', default=1000000, required=False, type=int, help='Max epoch to test, avoid overfitting'
    )
    parser.add_argument(
        '--corr', action='store_true', help='Whether to just train/test the correction module'
    )
    parser.add_argument(
        '--epochs', default=-1, type=int, required=False, help='Override max number of epochs'
    )
    parser.add_argument(
        '--no-tqdm', action='store_true', help='Disable tqdm'
    )
    args = parser.parse_args()
    if args.no_tqdm:
        from tqdm import tqdm
        from functools import partialmethod
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    config = get_exp_config(**vars(args))
    params = vars(args)
    params['config'] = deepcopy(config)
    run_task(params)
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    main()