import subprocess
import sys
from vrnntools.utils.cache_fpv import dataset_folds

if __name__ == '__main__':
    cmd = 'python run.py '
    cmd_args = ' '.join(sys.argv[1:])
    cmd += cmd_args
    fold_cmd =f'{cmd} --run-type trainval'
    print(f'\nRunning command: {fold_cmd}')
    subprocess.call(fold_cmd, shell=True)
    fold_cmd =f'{cmd} --run-type test'
    print(f'\nRunning command: {fold_cmd}')
    subprocess.call(fold_cmd, shell=True)