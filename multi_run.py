import subprocess
import sys
from vrnntools.utils.cache_fpv import dataset_folds

if __name__ == '__main__':
    cmd = 'python run.py '
    cmd_args = ' '.join(sys.argv[1:])
    cmd += cmd_args
    for fold in dataset_folds:
        fold_cmd =f'{cmd} --fold {fold}'
        print(f'\nRunning command: {fold_cmd}')
        subprocess.call(fold_cmd, shell=True)