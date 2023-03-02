import argparse
import numpy as np
import os
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', default='test_sgnet_jobs.txt', help='Which set of jobs to run.')
    parser.add_argument('--gpus', type=int, default=1, help='How many GPUs to run on')
    parser.add_argument('--cpus', type=int, default=12, help='How many CPUs to run on')
    parser.add_argument('--per-gpu', type=int, default=1, help='How many jobs per GPU')
    parser.add_argument('--chunk', type=int, default=0, help='Which chunk to take')
    parser.add_argument('--chunks', type=int, default=2, help='How many total chunks')
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--no-tqdm', action='store_true', help='Enable tqdm')
    args = parser.parse_args()
    jobs = args.jobs
    gpus = args.gpus
    cpus = args.cpus
    per_gpu = args.per_gpu
    chunk = args.chunk
    chunks = args.chunks
    use_cpu = args.use_cpu
    no_tqdm = args.no_tqdm

    assert os.path.exists(jobs), 'Jobs file missing'
    n_concurrent = int(gpus*per_gpu)
    cpus_per = cpus//n_concurrent
    with open(jobs, 'r') as f:
        lines = [x.strip() for x in f.readlines() if len(x) > 1]

    lines = list(np.array_split(lines, chunks)[chunk])
    splits = np.array_split(lines, n_concurrent)
    splits = [list(x) for x in splits]
    for i in range(n_concurrent):
        gpu_id = i//per_gpu
        cpu_min = i*cpus_per
        cpu_max = cpu_min + cpus_per - 1
        for k in range(len(splits[i])):
            if n_concurrent > 1:
                splits[i][k] = f'taskset -a --cpu-list {cpu_min}-{cpu_max} {splits[i][k]} --gpu-id {gpu_id}'
            else:
                splits[i][k] = f'{splits[i][k]}'
            if args.no_tqdm:
                splits[i][k] = f'{splits[i][k]} --no-tqdm'
            if args.use_cpu:
                splits[i][k] = f'{splits[i][k]} --use-cpu'
    splits = ['; '.join(x) for x in splits]
    splits = [f'({x})' for x in splits]
    all_splits = ' & '.join(splits)

    subprocess.call(all_splits, shell=True)
