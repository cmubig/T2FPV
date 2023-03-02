#!/bin/python3

import numpy as np
import sys
import os
import json
import argparse
import glob
from natsort import natsorted
import pandas as pd
import re
from matplotlib import pyplot as plt
import sys

def compute_best(path, label=None, max_epoch=-1, ade_norm=None, fde_norm=None, no_avg=False, fpv=False, test=False, multi=False, corr=False):
    input_file = '/trainval.log' if not corr else '/traincorr.log'
    if test:
        input_file = '/test.log' if not corr else '/testcorr.log'
    if not os.path.exists(path + input_file):
        lines = []
    else:
        with open(path + input_file, 'r') as f:
            lines = f.readlines()
    
    run_markers = []
    for i, line in enumerate(lines):
        if (not test and 'epoch: [0' in line) or (test and 'running checkpoint' in line):
            run_markers.append(i)
    if not len(run_markers):
        return None

    cur_run = lines[run_markers[-1]:]

    epoch = 0
    best_epoch = -1
    best_ade, best_fde = np.inf, np.inf
    best_ade_orig, best_fde_orig = np.inf, np.inf
    best_ade_med, best_fde_med = np.inf, np.inf
    if not multi:
        test_ade, test_fde = np.inf, np.inf
    else:
        test_ade, test_fde = {}, {}
        test_ade_med, test_fde_med = {}, {}
        test_ade_orig, test_fde_orig = {}, {}
    if max_epoch < 0:
        max_epoch = np.inf
    max_reached_epoch = -1
    max_possible_epoch = -1
    for line in cur_run:
        if 'Epoch' in line:
            epoch = int(line.split('[')[-1].split('/')[0])
            max_reached_epoch = max(max_reached_epoch, epoch)
            max_possible_epoch_ = int(line.split('[')[-1].split('/')[1].split(']')[0])
            max_possible_epoch = max(max_possible_epoch, max_possible_epoch_)
            if epoch > max_epoch:
                break
        if ('\teval: ' not in line) and ('\ttest' not in line):
            continue
        is_test = 'test' in line

        # Eval line
        if not is_test:
            ade = float(line.split(' MinADE: ')[-1].split(' ')[0])
            fde = float(line.split(' MinFDE: ')[-1].split(' ')[0])
            ade_med = float(line.split(' MinADEMed: ')[-1].split(' ')[0])
            fde_med = float(line.split(' MinFDEMed: ')[-1].split(' ')[0])
            if ade < best_ade:
                best_epoch = epoch
                best_ade = ade
                best_ade_med = ade_med
                best_fde = fde
                best_fde_med = fde_med
                if 'OrigMinADE' in line and 'OrigMinFDE' in line:
                    ade_orig = float(line.split(' OrigMinADE: ')[-1].split(' ')[0])
                    fde_orig = float(line.split(' OrigMinFDE: ')[-1].split(' ')[0])
                    best_ade_orig = ade_orig
                    best_fde_orig = fde_orig
        else:
            key = None if not multi else line.split('test')[-1].split('(')[-1].split(')')[0]
            assert key is not None, 'Multi must be enabled'
            if test:
                best_epoch = epoch
                max_reached_epoch = max_possible_epoch
            tmp_test_ade = float(line.split(' MinADE: ')[-1].split(' ')[0])
            tmp_test_fde = float(line.split(' MinFDE: ')[-1].split(' ')[0])
            tmp_test_ade_med = float(line.split(' MinADEMed: ')[-1].split(' ')[0])
            tmp_test_fde_med = float(line.split(' MinFDEMed: ')[-1].split(' ')[0])

            # Only test at end...
            test_ade[key] = tmp_test_ade
            test_fde[key] = tmp_test_fde
            test_ade_med[key] = tmp_test_ade_med
            test_fde_med[key] = tmp_test_fde_med
            if 'OrigMinADE' in line and 'OrigMinFDE' in line:
                tmp_test_ade_orig = float(line.split(' OrigMinADE: ')[-1].split(' ')[0])
                tmp_test_fde_orig = float(line.split(' OrigMinFDE: ')[-1].split(' ')[0])
                test_ade_orig[key] = tmp_test_ade_orig
                test_fde_orig[key] = tmp_test_fde_orig
    # No validation included for now
    test_names = natsorted([x for x in test_ade.keys()])
    data = { 'test_name': ['Val', *test_names],
            'ade': [best_ade, *[test_ade[k] for k in test_names]],
            'fde': [best_fde, *[test_fde[k] for k in test_names]],
            'ade_med': [best_ade_med, *[test_ade_med[k] for k in test_names]],
            'fde_med': [best_fde_med, *[test_fde_med[k] for k in test_names]],
            # 'ade_orig': [best_ade_orig, *[test_ade_orig[k] for k in test_names]],
            # 'fde_orig': [best_fde_orig, *[test_fde_orig[k] for k in test_names]],
            'det_ade': [0, *[0 for _ in range(len(test_names))]],
            'det_fde': [0, *[0 for _ in range(len(test_names))]],
            'ego_ade': [0, *[0 for _ in range(len(test_names))]],
            'ego_fde': [0, *[0 for _ in range(len(test_names))]],
            'AP1': [0, *[0 for _ in range(len(test_names))]],
            'mAP': [0, *[0 for _ in range(len(test_names))]],
            'best_epoch': [best_epoch]*(len(test_ade)+1),
            'max_epoch': [max_reached_epoch]*(len(test_ade)+1)}
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--corr', action='store_true')
    parser.add_argument('--out', default='./figures', type=str, help='Where to store output figures...')
    parser.add_argument('--ade-norm', default=None)
    parser.add_argument('--fde-norm', default=None)
    parser.add_argument('--no-avg', default=False, action='store_true')
    parser.add_argument('--no-fpv', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--no-multi', default=False, action='store_true')
    args = parser.parse_args()


    path = args.path
    no_avg = args.no_avg
    fpv = not args.no_fpv
    corr = args.corr
    multi = not args.no_multi
    assert fpv and multi, 'Non-FPV, non-multi not supported currently'
    test = args.test
    if os.path.isdir(path):
        compute_best(path, max_epoch = args.epoch, ade_norm=args.ade_norm, fde_norm=args.fde_norm, no_avg=no_avg, fpv=fpv, test=test, multi=multi, corr=corr)
        sys.exit(0)

    with open(path, 'r') as f:
        files = json.load(f)
    tag_info = {}
    all_res = []
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
            best_vals = np.inf*np.zeros((0, 8))
            for fold, path in zip(folds, matching_paths):
                fold_str = f'{fold}\t' if fold is not None else ''
                label = f'{fold_str}{item["key"]}'
                res = compute_best(path, label=label, max_epoch = args.epoch, 
                                ade_norm=args.ade_norm, fde_norm=args.fde_norm, no_avg=no_avg, fpv=fpv, test=test, multi=multi, corr=corr)
                if res is None or not len(res):
                    # File hasn't been created yet...
                    continue
                res['fold'] = fold
                res['algo'] = item['key']
                res['train_name'] = item['train']
                res = res[[*res.columns[-3:], *res.columns[:-3]]]
                all_res.append(res)
    # for res in all_res:
    #     print(res)
    #     print()
    all_res = pd.concat(all_res).reset_index(drop=True)
    if args.live:
        with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000, 'display.width', 300):
            # Look at trainval results...
            train_res = all_res[all_res.test_name == 'Val']
            #train_res = train_res[['fold', 'train_name', 'algo', 'ade', 'fde', 'ade_orig', 'fde_orig', 'best_epoch', 'max_epoch']]
            train_res = train_res[['fold', 'train_name', 'algo', 'ade', 'fde', 'best_epoch', 'max_epoch']]
            print(train_res)
            print()
            # print(train_res.groupby('algo').mean())
            # print()
            test_res = all_res[all_res.test_name != 'Val']
            if not len(test_res):
                sys.exit(0)
            print(test_res.groupby(['test_name', 'train_name', 'fold', 'algo']).mean())
            print()
            print(test_res.groupby(['test_name', 'train_name', 'algo']).mean())
            print()
            base_algos = []
            for i, row in test_res.iterrows():
                base_algos.append(row.algo.split(' ')[0])
            test_res['base_algo'] = base_algos
            dfs = []
            for i, group in test_res.groupby(['train_name', 'fold', 'base_algo']):
                base_res = group[group.algo == group.iloc[0].algo].reset_index(drop=True)
                other_res = group[group.algo != group.iloc[0].algo].reset_index(drop=True)
                base_res['ade_diff'] = 1
                base_res['fde_diff'] = 1
                other_res['ade_diff'] = other_res.ade/base_res.ade.iloc[0]
                other_res['fde_diff'] = other_res.fde/base_res.fde.iloc[0]
                dfs.extend([base_res, other_res])
            diff_df = pd.concat(dfs).reset_index(drop=True)
            # print('Improvement over baseline (higher is better)')
            # print(1 - diff_df.groupby(['test_name', 'train_name', 'algo', 'fold'])[['ade_diff', 'fde_diff']].mean().round(decimals=3))
            # print()
            # print(1 - diff_df.groupby(['test_name', 'train_name', 'algo'])[['ade_diff', 'fde_diff']].mean().round(decimals=3))
            # print()
            print('Raw ADE/FDE')
            #import pdb; pdb.set_trace()
            print(diff_df.groupby(['test_name', 'train_name', 'algo', 'fold'])[['ade', 'fde', 'ade_med', 'fde_med']].mean().round(decimals=3))
            print()
            print(diff_df.groupby(['test_name', 'train_name', 'algo'])[['ade', 'fde', 'ade_med', 'fde_med']].mean().round(decimals=3))
            sys.exit(0)
    # TODO: automatically determine this
    test_name_order = ['Val', 'FPV-GT', 'FPV-Noisy', 'FPV-Det', 'FPV-DetTrain']
    test_name_idx = [test_name_order.index(row.test_name) for _, row in all_res.iterrows()] 
    all_res['test_name_idx'] = test_name_idx
    fold_order = ['Zara1', 'Hotel', 'Zara2', 'Univ', 'ETH']
    fold_idx = [fold_order.index(row.fold) for _, row in all_res.iterrows()] 
    all_res['fold_idx'] = fold_idx
    all_res['original_idx'] = all_res.index
    all_res = all_res.sort_values(['fold_idx', 'train_name', 'test_name_idx', 'algo']).reset_index(drop=True)

    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000, 'display.width', 300):
        # See: https://stackoverflow.com/a/39566040
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


        print('Raw results:')
        print(all_res)
        #all_res = all_res[~all_res.algo.isin(['SGNet Det', 'SGNet CVAE'])]

        print('\n\n')
        print('Average performance per algorithm:')
        base_algos = []
        algo_ext = []
        algo_ext2 = []
        for i, row in all_res.iterrows():
            base_algos.append(row.algo.split(' ')[0])
            algo_ext.append('Algo' + row.algo[len(row.algo.split(' ')[0]):])
            algo_ext2.append(row.algo[len(row.algo.split(' + ')[0])+3:])
        all_res['base_algo'] = base_algos
        all_res['algo_ext'] = algo_ext
        all_res['algo_ext2'] = algo_ext2
        model_groups = all_res.groupby(['train_name', 'base_algo', 'algo_ext', 'algo_ext2', 'algo', 'fold_idx', 'fold', 'test_name_idx', 'test_name'])
        means = model_groups[['ade', 'fde', 'ade_med', 'fde_med', 'ego_ade', 'ego_fde', 'det_ade', 'det_fde', 'AP1', 'mAP', 'best_epoch', 'max_epoch', 'original_idx']].mean()
        combined = pd.concat([means], axis=1)
        print(combined)

        no_val = all_res[all_res.test_name != 'Val']
        corrs = []
        imp = []
        for _, row in no_val.algo_ext2.items():
            corrs.append('Yes' if 'Corr' in row else 'No')
            if row == '' or row == 'Corr':
                imp.append('Linear-interp')
            elif 'Smooth' in row:
                imp.append('Smooth')
            elif 'NAOMI' in row:
                imp.append('NAOMI')
            else:
                imp.append('(Unknown)')
        no_val['corrs'] = corrs
        no_val['imp'] = imp
        dfs = []
        for i, group in no_val.groupby(['train_name', 'fold', 'base_algo', 'imp']):
            # base_res = group[group.algo == group.iloc[0].algo].reset_index(drop=True)
            # other_res = group[group.algo != group.iloc[0].algo].reset_index(drop=True)
            base_res = group[group.corrs == 'No'].reset_index(drop=True)
            other_res = group[group.corrs != 'No'].reset_index(drop=True)
            base_res['ade_diff'] = 1
            base_res['fde_diff'] = 1
            base_res['ade_med_diff'] = 1
            base_res['fde_med_diff'] = 1
            other_res['ade_diff'] = other_res.ade/base_res.ade.iloc[0]
            other_res['fde_diff'] = other_res.fde/base_res.fde.iloc[0]
            other_res['ade_med_diff'] = other_res.ade_med/base_res.ade_med.iloc[0]
            other_res['fde_med_diff'] = other_res.fde_med/base_res.fde_med.iloc[0]
            dfs.extend([base_res, other_res])
        no_val = pd.concat(dfs).reset_index(drop=True)
        print('\n\n')
        print('Average performance across folds:')
        # Idx columns
        model_groups = no_val.groupby(['train_name', 'algo', 'base_algo', 'algo_ext', 'algo_ext2', 'imp', 'corrs', 'test_name_idx', 'test_name'])
        folds = model_groups['fold'].unique()
        # Columns to take means of
        means = model_groups[['ade', 'fde', 'ade_med', 'fde_med', 'ego_ade', 'ego_fde', 'det_ade', 'det_fde', 'best_epoch', 'max_epoch', 'ade_diff', 'fde_diff', 'original_idx']].mean()
        combined = pd.concat([means, folds], axis=1)
        test_combined = combined
        print(combined)
        ade_ax = means.reset_index().set_index('test_name').groupby('algo')['ade'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('ADE per Test Set (avg. over folds)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('ADE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_test.png'), dpi=300)
        plt.clf()

        # Average performance (over all folds), x-axis = train_name
        ade_ax = means.reset_index().set_index('train_name').groupby('algo')['ade'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('ADE per Train Set (avg. over folds + tests)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('ADE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_train.png'), dpi=300)
        plt.clf()

        bar_df = means.reset_index()
        bar_df = bar_df.sort_values('original_idx').reset_index(drop=True)
        #bar_df = bar_df[bar_df.algo != bar_df.base_algo]

        # for k, v in citations.items():
        #     tex_out = tex_out.replace(k, k[:len(k)-1]+'~\\cite{' + v + '} ')
        # corrs = []
        # imp = []
        # for _, row in bar_df.algo_ext2.items():
        #     corrs.append('Yes' if 'Corr' in row else 'No')
        #     if row == '' or row == 'Corr':
        #         imp.append('Linear-interp')
        #     elif 'Smooth' in row:
        #         imp.append('Smooth')
        #     elif 'NAOMI' in row:
        #         imp.append('NAOMI')
        # bar_df['corrs'] = corrs
        # bar_df['imp'] = imp

        #bar_df['Approach'] = bar_df['algo_ext2']
        bar_df['Approach'] = bar_df['base_algo']

        algo_names = []
        for _, row in bar_df.iterrows():
            name = row.imp
            if row.corrs == 'Yes':
                name += ' + Corr'
            algo_names.append(name)
        bar_df['algo_name'] = algo_names
        col_order = bar_df.algo_name.unique()
        bar_colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green', 'tab:orange', 'tab:orange']
        alphas = [1, 0.3, 1, 0.3, 1, 0.3]
        pivot_df = bar_df.pivot_table(index='Approach', columns='algo_name', values='ade', sort=False)[col_order] 
        tmp_ax = pivot_df.plot(kind='bar', color=bar_colors)
        for bar_algo, alpha, bar_color in zip(tmp_ax.containers, alphas, bar_colors):
            for bar in bar_algo:
                bar.set_alpha(alpha)
                bar.set_edgecolor('k')
        plt.title('ADE Values (avg. over folds)')
        plt.legend(pivot_df.columns, title=pivot_df.columns.name)
        #plt.tick_params(axis=u'x', which=u'both', lenght=0)
        plt.xlabel('')
        plt.xticks(rotation='horizontal')
        plt.ylabel('ADE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_bar.png'), dpi=300)

        tmp_ax = bar_df.pivot_table(index='Approach', columns='algo_name', values='fde', sort=False)[col_order].plot(kind='bar', color=bar_colors)
        for bar_algo, alpha, bar_color in zip(tmp_ax.containers, alphas, bar_colors):
            for bar in bar_algo:
                bar.set_alpha(alpha)
                bar.set_edgecolor(bar_color)
        plt.title('FDE Values (avg. over folds)')
        #plt.tick_params(axis=u'x', which=u'both', lenght=0)
        plt.xlabel('')
        plt.xticks(rotation='horizontal')
        plt.ylabel('FDE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_bar.png'), dpi=300)

        bar_df = bar_df[bar_df.corrs == 'Yes']
        col_order = bar_df.imp.unique()
        bar_df['ade_diff_p'] = (1 - bar_df['ade_diff'])*100
        tmp_ax = bar_df.pivot_table(index='Approach', columns='imp', values='ade_diff_p', sort=False)[col_order].plot(kind='bar')
        plt.title('ADE Improvement with Corr (avg. over folds)')
        #plt.tick_params(axis=u'x', which=u'both', lenght=0)
        plt.xlabel('')
        plt.xticks(rotation='horizontal')
        #import pdb; pdb.set_trace()
        y1, y2 = plt.gca().get_ylim()
        plt.ylim(0, y2+9)
        plt.ylabel('ADE Percent Improvement')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_bar_diff.png'), dpi=300)
        import pdb; pdb.set_trace()

        bar_df['fde_diff_p'] = (1 - bar_df['fde_diff'])*100
        tmp_ax = bar_df.pivot_table(index='Approach', columns='imp', values='fde_diff_p', sort=False)[col_order].plot(kind='bar')
        plt.title('FDE Improvement with Corr (avg. over folds)')
        #plt.tick_params(axis=u'x', which=u'both', lenght=0)
        plt.xlabel('')
        plt.xticks(rotation='horizontal')
        # Same lim as above
        plt.ylim(0, y2+9)
        plt.ylabel('FDE Percent Improvement')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_bar_diff.png'), dpi=300)

        fde_ax = means.reset_index().set_index('train_name').groupby('algo')['fde'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('FDE per Train Set (avg. over folds + tests)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('FDE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_train.png'), dpi=300)
        plt.clf()

        # Average performance (over all folds), x-axis = train_name
        ade_ax = means.reset_index().set_index('train_name').groupby('algo')['ade_med'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('ADE Median per Train Set (avg. over folds + tests)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('ADE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_med_train.png'), dpi=300)
        plt.clf()

        fde_ax = means.reset_index().set_index('train_name').groupby('algo')['fde_med'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('FDE Median per Train Set (avg. over folds + tests)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('FDE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_med_train.png'), dpi=300)
        plt.clf()

        ade_ax = means.reset_index().set_index('train_name').groupby('algo')['ade_diff'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('ADE Diff per Train Set (avg. over folds + tests)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('ADE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_train_diff.png'), dpi=300)
        plt.clf()

        fde_ax = means.reset_index().set_index('train_name').groupby('algo')['fde_diff'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('FDE Diff per Train Set (avg. over folds + tests)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('FDE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_train_diff.png'), dpi=300)
        plt.clf()


        fde_ax = means.reset_index().set_index('test_name').groupby('algo')['fde'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('FDE per Test Set (avg. over folds)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('FDE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_test.png'), dpi=300)
        plt.clf()

        print('\n\n')


        print('Average performance across test sets:')
        model_groups = no_val.groupby(['train_name', 'algo', 'base_algo', 'fold'])
        sets = model_groups['test_name'].unique()
        means = model_groups[['ade', 'fde', 'ego_ade', 'ego_fde', 'det_ade', 'det_fde', 'ade_med', 'fde_med', 'best_epoch', 'max_epoch', 'original_idx']].mean()
        combined = pd.concat([means, sets], axis=1)
        fold_combined = combined
        print(combined)
        ade_ax = means.reset_index().set_index('fold').groupby('algo')['ade'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('ADE per Fold (avg. over test sets)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('ADE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_fold.png'), dpi=300)
        plt.clf()

        fde_ax = means.reset_index().set_index('fold').groupby('algo')['fde'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('FDE per Fold (avg. over test sets)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('FDE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_fold.png'), dpi=300)
        plt.clf()

        ade_ax = means.reset_index().set_index('fold').groupby('algo')['ade_med'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('ADE per Fold (avg. over test sets)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('ADE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'ade_med_fold.png'), dpi=300)
        plt.clf()

        fde_ax = means.reset_index().set_index('fold').groupby('algo')['fde_med'].plot(legend=True, marker='o', alpha=0.5)
        plt.title('FDE per Fold (avg. over test sets)')
        plt.tick_params(axis=u'x', which=u'both',length=0)
        plt.xlabel('')
        plt.ylabel('FDE (meters)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'fde_med_fold.png'), dpi=300)
        plt.clf()


        print('\n\n')


        print('Average performance across test sets & folds:')
        model_groups = no_val.groupby(['train_name', 'algo'])
        sets = model_groups['test_name'].unique()
        folds = model_groups['fold'].unique()
        means = model_groups[['ade', 'fde', 'ego_ade', 'ego_fde', 'det_ade', 'det_fde', 'best_epoch', 'max_epoch', 'original_idx']].mean()
        combined = pd.concat([means, sets, folds], axis=1)
        print(combined)
        print('\n\n')

        print('Average performance across test sets, folds, & algorithms:')
        overall = all_res.groupby('test_name')[['ade', 'fde', 'mAP', 'AP1']].mean()
        print(overall)
        print(1 - (overall.iloc[0]/overall))
        overall['ade_inc_abs'] = overall['ade'] - overall['ade'].iloc[0]
        overall['ade_inc_%'] = overall['ade_inc_abs'] / overall['ade'].iloc[0]
        overall['fde_inc_abs'] = overall['fde'] - overall['fde'].iloc[0]
        overall['fde_inc_%'] = overall['fde_inc_abs'] / overall['fde'].iloc[0]
        overall['mAP_dec_abs'] = -(overall['mAP'] - overall['mAP'].iloc[0])
        overall['mAP_dec_%'] = (overall['mAP_dec_abs'] / overall['mAP'].iloc[0])
        print(overall)

        # print('\n\n')
        # if 'SGNet CVAE + LN' in all_res.algo.values:
        #     print("best perf: SGNET CVAE + LN")
        #     best_algo = all_res[all_res.algo == 'SGNet CVAE + LN'].groupby('test_name')[['ade', 'fde', 'mAP', 'AP1']].mean()
        #     best_algo['ade_inc_abs'] = best_algo['ade'] - best_algo['ade'].iloc[0]
        #     best_algo['ade_inc_%'] = best_algo['ade_inc_abs'] / best_algo['ade'].iloc[0]
        #     best_algo['fde_inc_abs'] = best_algo['fde'] - best_algo['fde'].iloc[0]
        #     best_algo['fde_inc_%'] = best_algo['fde_inc_abs'] / best_algo['fde'].iloc[0]
        #     best_algo['mAP_dec_abs'] = -(best_algo['mAP'] - best_algo['mAP'].iloc[0])
        #     best_algo['mAP_dec_%'] = (best_algo['mAP_dec_abs'] / best_algo['mAP'].iloc[0])
        #     print(best_algo)
        #     print('\n\n')

        # TODO: more analyses?
        def to_latex(df):
            markers = ['VRNN', 'A-VRNN', 'SGNet']
            tex_out = df.to_latex(index_names=False, index=False)
            #tex_out = tex_out.replace('\\toprule\n', '\\hline\n').replace('\\midrule\n', '').replace('\\bottomrule\n', '\\hline\n')
            tex_out = tex_out.replace('\\midrule\n', '\n')
            tex_out = tex_out.replace('tabular}{lllllll', 'tabular}{l|ccccc|c')
            tex_out = tex_out.replace('\\textbackslash textbf\\{', '\\textbf{')
            tex_out = tex_out.replace('\\}', '}')
            tex_out = re.sub(' +', ' ', tex_out)
            for marker in markers:
                tex_out = tex_out.replace(f'\n {marker} &', f'\n\\midrule\n {marker} &')
            # Order of replacement matters, to ensure acvrnn doesn't override vrnn
            citations = {'A-VRNN ': 'acvrnn', 'VRNN ': 'vrnn', 'SGNet ': 'sgnet', 
                         'NAOMI ': 'naomi', 'Smooth ': 'retrack'}
            for k, v in citations.items():
                tex_out = tex_out.replace(k, k[:len(k)-1]+'~\\cite{' + v + '} ')
            # \rowcolor{lightgray}
            tex_lines = tex_out.split('\n')
            n_entries = 0
            #tex_lines = ['\n\\rowcolor{lightgray}\n ' + x if i % 2 == 0 and i > 1 else x for i, x in enumerate(tex_lines)]
            for i, x in enumerate(tex_lines):
                if x.strip().split(' ')[0] in markers:
                    n_entries += 1
                    if n_entries % 2:
                        tex_lines[i] = '\\rowcolor{lightgray}\n' + x
            tex_out = '\n'.join(tex_lines)
            return tex_out

        sus = fold_combined[['ade', 'fde', 'original_idx']].reset_index().drop(columns=['train_name'])
        sus2 = sus.groupby(['algo', 'base_algo']).mean().reset_index()
        sus2['fold'] = 'Avg'
        sus2 = sus2[['algo', 'base_algo', 'fold', 'ade', 'fde', 'original_idx']]
        sus = pd.concat([sus, sus2]).reset_index(drop=True)
        cols = ['Algorithm', *sus.fold.unique()]
        sus_df = {col: [] for col in cols}
        sus_ade = {}
        sus_fde = {}
        for base_algo in sus.base_algo.unique():
            base_ade = {fold: sus[(sus.fold == fold) & (sus.base_algo == base_algo)].ade.min() for fold in sus.fold.unique()}
            base_fde = {fold: sus[(sus.fold == fold) & (sus.base_algo == base_algo)].fde.min() for fold in sus.fold.unique()}
            sus_ade[base_algo] = base_ade
            sus_fde[base_algo] = base_fde
        for algo_name, algo_df in sus.groupby('algo'):
            sus_df['Algorithm'].append(algo_name)
            for _, algo_row in algo_df.iterrows():
                base_algo = algo_row.base_algo
                tmp_ade = f'{algo_row.ade:.2f}'
                tmp_fde = f'{algo_row.fde:.2f}'
                if tmp_ade == f'{sus_ade[base_algo][algo_row.fold]:.2f}':
                    tmp_ade = '\\textbf{' + tmp_ade + '}'
                if tmp_fde == f'{sus_fde[base_algo][algo_row.fold]:.2f}':
                    tmp_fde = '\\textbf{' + tmp_fde + '}'
                err_str = f'{tmp_ade} / {tmp_fde}'
                sus_df[algo_row.fold].append(err_str)
        for k, v in sus_df.items():
            v = [x for _, x in sorted(zip(sus2.original_idx, v))]
            sus_df[k] = v
        sus_df = pd.DataFrame(sus_df)
        # Hard code order
        # new_order = [6, 8, 7, 0, 2, 1, 3, 5, 4]
        # print(to_latex(sus_df.iloc[new_order].reset_index(drop=True)))
        print(to_latex(sus_df.reset_index(drop=True)))
        #print(to_latex(sus_df.reset_index(drop=True)))
        print('\n\n')



        sus = test_combined[['ade', 'fde']].reset_index().drop(columns=['train_name'])
        sus2 = sus.groupby('algo').mean().reset_index()
        sus2['test_name'] = 'Avg'
        sus2 = sus2[['algo', 'test_name', 'ade', 'fde']]
        sus = pd.concat([sus, sus2]).reset_index(drop=True)
        cols = ['Algorithm', *sus.test_name.unique()]
        sus_df = {col: [] for col in cols}
        sus_ade = {test_name: sus[sus.test_name == test_name].ade.min() for test_name in sus.test_name.unique()}
        sus_fde = {test_name: sus[sus.test_name == test_name].fde.min() for test_name in sus.test_name.unique()}
        for algo_name, algo_df in sus.groupby('algo'):
            sus_df['Algorithm'].append(algo_name)
            for _, algo_row in algo_df.iterrows():
                tmp_ade = f'{algo_row.ade:.2f}'
                tmp_fde = f'{algo_row.fde:.2f}'
                if tmp_ade == f'{sus_ade[algo_row.test_name]:.2f}':
                    tmp_ade = '\\textbf{' + tmp_ade + '}'
                if tmp_fde == f'{sus_fde[algo_row.test_name]:.2f}':
                    tmp_fde = '\\textbf{' + tmp_fde + '}'
                err_str = f'{tmp_ade}/{tmp_fde}'
                sus_df[algo_row.test_name].append(err_str)
        sus_df = pd.DataFrame(sus_df)
        # print(to_latex(sus_df))
        # print('\n\n')


