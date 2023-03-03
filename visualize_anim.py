# ------------------------------------------------------------------------------
# @file:    visualize.py
# @brief:   This file contains the implementation of visualization
# @author:  Meghdeep Jana
# @date:    Last modified on February 20th, 2023
# ------------------------------------------------------------------------------

import numpy as np
# import torch
import matplotlib.pyplot as plt
import os
# from vrnntools.utils import visualization
from natsort import natsorted
import argparse

min_batches = 0
max_batches = 50

allowed = set([(42, 19)])

class Visualizer:
    def __init__(self, output_dir, corr = False, smooth = False, hist_len = 8, algo_name='', fold_name='', algo_out_name='', imp_out_name='', plot_dir='vis_out'):
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        self.hist_len = hist_len
        self.smooth = smooth
        self.algo_name = algo_name
        self.algo_out_name = algo_out_name
        self.fold_name = fold_name
        self.imp_out_name = imp_out_name
        self.corr = corr
        self.fold = None
        self.run = None
        self.epoch = None
        self.batch = None
        self.data = None

        os.makedirs(self.plot_dir, exist_ok=True)

    def load_data(self, fold, run, epoch, run_base, run_imp, get_cutoffs = False, to_render=None):
        self.fold = fold
        self.run = run
        self.run_base = run_base
        self.run_imp = run_imp
        self.imp_name = 'NAOMI' if 'naomi' in run_imp else 'Smooth' if 'smooth' in run_imp else 'Linear-interp'
        if epoch is None:
            epoch = os.listdir(f'{self.output_dir}{fold}/{run}/trajs')[0]
            epoch = int(epoch.split('epoch')[-1].split('_batch')[0])

        base_epoch = os.listdir(f'{self.output_dir}{fold}/{run_base}/trajs')[0]
        base_epoch = int(base_epoch.split('epoch')[-1].split('_batch')[0])
        base_batches = os.listdir(f"{self.output_dir}{fold}/{run}/trajs")
        base_batches = natsorted([int(x.split('batch')[-1].split('.npy')[0]) for x in base_batches])

        imp_epoch = os.listdir(f'{self.output_dir}{fold}/{run_imp}/trajs')[0]
        imp_epoch = int(imp_epoch.split('epoch')[-1].split('_batch')[0])
        imp_batches = os.listdir(f"{self.output_dir}{fold}/{run}/trajs")
        imp_batches = natsorted([int(x.split('batch')[-1].split('.npy')[0]) for x in imp_batches])

        self.epoch = epoch
        batches = os.listdir(f"{self.output_dir}{fold}/{run}/trajs")
        batches = natsorted([int(x.split('batch')[-1].split('.npy')[0]) for x in batches])
        assert batches == base_batches, 'Mismatch'
        assert batches == imp_batches, 'Mismatch'
        if get_cutoffs:
            cutoffs = []
            for batch in batches:
                if batch < min_batches:
                    break
                if batch > max_batches:
                    break
                self.batch = batch
                base_file_path = f"{self.output_dir}{fold}/{run_base}/trajs/test_epoch{base_epoch}_batch{batch}.npy"
                imp_file_path = f"{self.output_dir}{fold}/{run_imp}/trajs/test_epoch{imp_epoch}_batch{batch}.npy"
                file_path = f"{self.output_dir}{fold}/{run}/trajs/test_epoch{epoch}_batch{batch}.npy"
                self.base_data = np.load(base_file_path, allow_pickle=True).item()
                self.imp_data = np.load(imp_file_path, allow_pickle=True).item()
                self.data = np.load(file_path, allow_pickle=True).item()
                cutoffs.extend([(batch, agent_idx) for agent_idx in self.plot_data(batch, get_cutoffs)])
            return cutoffs
        else:
            for batch in batches:
                if batch > max_batches:
                    break
                if batch < min_batches:
                    break
                self.batch = batch
                base_file_path = f"{self.output_dir}{fold}/{run_base}/trajs/test_epoch{base_epoch}_batch{batch}.npy"
                imp_file_path = f"{self.output_dir}{fold}/{run_imp}/trajs/test_epoch{imp_epoch}_batch{batch}.npy"
                file_path = f"{self.output_dir}{fold}/{run}/trajs/test_epoch{epoch}_batch{batch}.npy"
                self.base_data = np.load(base_file_path, allow_pickle=True).item()
                self.imp_data = np.load(imp_file_path, allow_pickle=True).item()
                self.data = np.load(file_path, allow_pickle=True).item()
                self.plot_data(batch, get_cutoffs, to_render)

    def plot_data(self, batch, get_cutoffs, to_render=None):
        # SMALL_SIZE = 14
        # MEDIUM_SIZE = 16
        # BIGGER_SIZE = 18

        # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # for agent in range(self.data['fut_abs'].shape[1]):
        cut_offs = []
        for agent in range(self.data['fut_abs'].shape[1]):
            if allowed is not None and (batch, agent) not in allowed:
                continue
            # n_agree = 3
            # if (self.data['hist_abs'][:n_agree, agent] != self.data['hist_abs_corr'][:n_agree, agent]).all():
            #     continue
            # if (self.data['hist_abs'][:, agent] == self.data['hist_abs_corr'][:, agent]).all():
            #     continue
            # Expects 2d inputs
            if to_render is not None:
                if (batch, agent) not in to_render:
                    continue
            def mse(a, b):
                return np.sum(np.sqrt(np.sum((a - b) ** 2, axis=-1)), axis=0)/len(a)
            def mse_i(a, b, i):
                return np.sqrt(np.sum((a - b) ** 2, axis=-1))[i]
            gt = self.data['gt_abs'][:8, agent]
            orig = self.base_data['hist_abs'][:8, agent]
            corr = self.data['hist_abs_corr'][:8, agent]
            imp_field_name = 'hist_abs_naomi' if self.imp_name == 'NAOMI' else 'hist_abs_smooth' if self.imp_name == 'Smooth' else 'hist_abs'
            imp = self.imp_data[imp_field_name][:8, agent]
            fut_gt = self.data['gt_abs'][8:, agent]
            fut_orig = self.base_data['fut_abs'][:, agent]
            fut_corr = self.data['fut_abs'][:, agent]
            fut_imp = self.imp_data['fut_abs'][:, agent]
            assert (self.data['gt_abs'] == self.base_data['gt_abs']).all(), 'Mismatch in base vs. corr'
            mse_orig = mse(gt, orig)
            mse_corr = mse(gt, corr)
            mse_imp = mse(gt, imp)
            mse0_orig = mse_i(gt, orig, 0)
            mse0_corr = mse_i(gt, corr, 0)
            mse0_imp = mse_i(gt, imp, 0)
            mse7_orig = mse_i(gt, orig, 7)
            mse7_corr = mse_i(gt, corr, 7)
            mse7_imp = mse_i(gt, imp, 7)
            ade_orig = mse(fut_gt, fut_orig)
            ade_corr =  mse(fut_gt, fut_corr)
            ade_imp =  mse(fut_gt, fut_imp)
            fde_orig = mse_i(fut_gt, fut_orig, -1)
            fde_corr =  mse_i(fut_gt, fut_corr, -1)
            fde_imp =  mse_i(fut_gt, fut_imp, -1)
            gt_rel = np.zeros_like(self.data['gt_abs'][:, agent])
            gt_rel[1:] = self.data['gt_abs'][:, agent][1:] - self.data['gt_abs'][:, agent][0:-1]
            gt_dist = np.sum(np.sqrt(np.sum(gt_rel**2, axis=-1)), axis=0)

            # Ensure the ground truth actually moves somewhere
            if gt_dist < 1:
                continue
            # Ensures a decent match between ground truth and original (i.e. from Hungarian algorithm)
            if mse0_corr >= 1.0:
                continue

            # Want following partial order on final input traj point: gt < corr < imp, orig
            # if mse7_orig <= mse7_corr or mse7_imp <= mse7_corr:
            #     continue
            # if mse_orig <= mse_corr or mse_imp <= mse_corr:
            #     continue
            if not (mse7_corr < mse7_imp):
                continue
            if not (mse_corr < mse_imp):
                continue

            # Want the following partial order of errors: gt < corr < imp < orig
            if not (fde_corr < fde_imp):
                continue
            if not (ade_corr < ade_imp):
                continue
            # if ade_orig <= ade_corr or fde_orig <= fde_corr:
            #     continue
            # if ade_imp <= ade_corr or fde_imp <= fde_corr:
            #     continue
            # if ade_orig <= ade_imp or fde_orig <= fde_imp:
            #     continue
            cut_offs.append(agent)
        if get_cutoffs:
            return cut_offs

        imp_color_idx = 6 if self.imp_out_name == 'linear' else 8 if self.imp_out_name == 'naomi' else 2
        base_color_index = [14, 4, imp_color_idx, 8]         # black, green, {red, purple, orange}, green (tab20)

        # Adjust brightness if the colors are too/less bright (in {0, 1, 2, 3})
        face_brightness = 0
        face_color_index = [i + face_brightness for i in base_color_index]

        edge_brightness = 0
        edge_color_index = [i + edge_brightness for i in base_color_index]

        edgecolors = plt.colormaps['tab20'](edge_color_index)[:, :3]
        colors = plt.colormaps['tab20'](face_color_index)[:, :3]

        # Only scatter plot supports hatch
        default_hatch = 10 * '/'
        default_kwargs = {'zorder': 100, 'alpha': 1}  # for scatter


        # For plt.scatter
        hist_args_list = [
            {'facecolor': 'w', 'edgecolor': edgecolors[0], 'marker': 'o', 's': 30},
            {'facecolor': 'w', 'edgecolor': edgecolors[1], 'marker': 'D', 's': 30},
            {'facecolor': 'w', 'edgecolor': edgecolors[2], 'marker': 's', 's': 30},
            {'facecolor': 'w', 'edgecolor': edgecolors[3], 'marker': 'p', 's': 40},
        ]
        pred_args_list = [
            {'facecolor': colors[0], 'edgecolor': colors[0], 'marker': 'o', 's': 30},
            {'facecolor': colors[1], 'edgecolor': colors[1], 'marker': 'D', 's': 30},
            {'facecolor': colors[2], 'edgecolor': colors[2], 'marker': 's', 's': 30},
            {'facecolor': colors[3], 'edgecolor': colors[3], 'marker': 'p', 's': 40},
        ]

        # For plt.plot
        lgd_args_list = [
            {'color': colors[0], 'markerfacecolor': colors[0], 'markeredgecolor': colors[0], 'linewidth': 1, 'marker': 'o', 'markersize': 5},
            {'color': colors[1], 'markerfacecolor': colors[1], 'markeredgecolor': colors[1], 'linewidth': 1, 'marker': 'D', 'markersize': 5},
            {'color': colors[2], 'markerfacecolor': colors[2], 'markeredgecolor': colors[2], 'linewidth': 1, 'marker': 's', 'markersize': 5},
            {'color': colors[3], 'markerfacecolor': colors[3], 'markeredgecolor': colors[3], 'linewidth': 1, 'marker': 'p', 'markersize': 6},
        ]


        plt.figure(figsize=(4, 4))

        for agent in cut_offs:
            if to_render is not None:
                if (batch, agent) not in to_render:
                    continue
            full_base = np.concatenate([self.base_data['hist_abs'][:, agent], self.base_data['fut_abs'][:, agent]])
            full_corr = np.concatenate([self.data['hist_abs_corr'][:, agent], self.data['fut_abs'][:, agent]])

            imp_field_name = 'hist_abs_naomi' if self.imp_name == 'NAOMI' else 'hist_abs_smooth' if self.imp_name == 'Smooth' else 'hist_abs'
            full_imp = np.concatenate([self.imp_data[imp_field_name][:, agent], self.imp_data['fut_abs'][:, agent]])
            full_gt = self.data['gt_abs'][:, agent]

            hist_valid = self.data['hist_valid'][:, agent]
            imputed = hist_valid == 0

            ## 1. Draw lines for all (history + pred)
            #for ax in axs:
            #plt.plot(full_corr[:,1], full_corr[:,0],           '-', **lgd_args_list[1])
            #plt.plot(full_imp[:,1], full_imp[:,0],             '-', **lgd_args_list[2])
            # plt.plot(full_base[:,1], full_base[:,0],           '-', **lgd_args_list[3])

            def animate(i):
                plt.clf()
                plt.plot(full_gt[:,1], full_gt[:,0],               '-', **lgd_args_list[0])
                plt.scatter(full_gt[:self.hist_len,1][~imputed], full_gt[:self.hist_len,0][~imputed],              **hist_args_list[0], **default_kwargs, hatch=default_hatch)
                plt.scatter(full_gt[:self.hist_len,1][imputed], full_gt[:self.hist_len,0][imputed],                **hist_args_list[0], **default_kwargs)
                plt.scatter(full_gt[self.hist_len:,1], full_gt[self.hist_len:,0],          **pred_args_list[0], **default_kwargs)

                # plt.scatter(full_base[:self.hist_len,1][~imputed], full_base[:self.hist_len,0][~imputed],          **hist_args_list[3], **default_kwargs, hatch=default_hatch)

                # plt.scatter(full_base[:self.hist_len,1][imputed], full_base[:self.hist_len,0][imputed],            **hist_args_list[3], **default_kwargs)

                plt.plot(full_imp[:(i), 1], full_imp[:(i), 0], '-', **lgd_args_list[2])
                scatter_idx = min(i, self.hist_len)
                plt.scatter(full_imp[:scatter_idx,1][~imputed[:scatter_idx]], full_imp[:scatter_idx,0][~imputed[:scatter_idx]],          **hist_args_list[2], **default_kwargs, hatch=default_hatch)
                plt.scatter(full_imp[:scatter_idx, 1][imputed[:scatter_idx]], full_imp[:scatter_idx,0][imputed[:scatter_idx]],            **hist_args_list[2], **default_kwargs)
                if i > self.hist_len:
                    plt.scatter(full_imp[self.hist_len:i,1], full_imp[self.hist_len:i,0],      **pred_args_list[2], **default_kwargs)
                plt.plot(full_corr[:(i), 1], full_corr[:(i), 0], '-', **lgd_args_list[1])
                scatter_idx = min(i, self.hist_len)
                plt.scatter(full_corr[:scatter_idx,1][~imputed[:scatter_idx]], full_corr[:scatter_idx,0][~imputed[:scatter_idx]],          **hist_args_list[1], **default_kwargs, hatch=default_hatch)
                plt.scatter(full_corr[:scatter_idx, 1][imputed[:scatter_idx]], full_corr[:scatter_idx,0][imputed[:scatter_idx]],            **hist_args_list[1], **default_kwargs)
                if i > self.hist_len:
                    plt.scatter(full_corr[self.hist_len:i,1], full_corr[self.hist_len:i,0],      **pred_args_list[1], **default_kwargs)
                ## 4. Draw dummy for legend
                plt.plot(np.nan, np.nan,      **lgd_args_list[0], label='GT')
                plt.plot(np.nan, np.nan,      **lgd_args_list[1], label=f'{self.imp_name} + CoFE (Ours)')
                plt.plot(np.nan, np.nan,      **lgd_args_list[2], label=self.imp_name)
                # plt.plot(np.nan, np.nan,      **lgd_args_list[3], label='Linear Interp.')


                handles, labels = plt.gca().get_legend_handles_labels()
                #specify order of items in legend
                # order = [0, 3, 2, 1]
                order = [0, 2, 1]
                #add legend to plot
                plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=2)

                # plt.axis('scaled')
                # x1, x2 = plt.gca().get_xlim()
                # y1, y2 = plt.gca().get_ylim()

                # all_traj = np.concatenate([full_gt, full_corr, full_imp, full_base], axis=0)
                all_traj = np.concatenate([full_gt, full_corr, full_imp], axis=0)
                y_min, x_min = all_traj.min(axis=0)
                y_max, x_max = all_traj.max(axis=0)

                x_min = np.floor(x_min)
                y_min = np.floor(y_min)
                x_max = np.ceil(x_max)
                y_max = np.ceil(y_max)

                height = y_max - y_min
                width = x_max - x_min

                if height > width:
                    x_min -= (height - width) / 2
                    x_max += (height - width) / 2
                elif height < width:
                    y_min -= (width - height) / 2
                    y_max += (width - height) / 2

                plt.xlim([x_min, x_max])
                plt.ylim([y_min, y_max])


                tick_r = 1.0
                xtick_vals = np.arange(x_min, x_max, tick_r)
                plt.xticks(xtick_vals)

                ytick_vals = np.arange(y_min, y_max, tick_r)
                plt.yticks(ytick_vals)

                plt.tick_params(
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    right=False,
                    left=False,
                    labelleft=False,
                    labelright=False,
                    labelbottom=False) # labels along the bottom edge are off
                plt.grid(True, which='both', alpha=0.5)
                ax = plt.gca()

                ax2 = ax.twinx()
                ax2.scatter(np.NaN, np.NaN, **hist_args_list[0], **default_kwargs, label='Observation', hatch=default_hatch)
                ax2.scatter(np.NaN, np.NaN, **hist_args_list[0], **default_kwargs, label='Missing (Imputed)')
                ax2.scatter(np.NaN, np.NaN, **pred_args_list[0], **default_kwargs, label='Prediction')

                ax2.set_yticks([])
                ax2.get_yaxis().set_visible(False)
                ax2.legend(loc=3)

                for tick in ax.yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

                # if self.smooth:
                #     plt.plot(self.data['hist_abs_smooth'][:,agent,0], self.data['hist_abs_smooth'][:,agent,1], 'm')
                #     plt.legend(['hist_gt', 'hist_abs', 'fut_gt' ,'fut_abs', 'start_gt', 'start_abs', 'hist_abs_smooth'])
                # plt.title(f"{self.algo_name} {self.fold_name} Predictions")
                plt.tight_layout()

            from matplotlib.animation import FuncAnimation
            anim = FuncAnimation(plt.gcf(), animate, frames=21, interval=400, repeat=False)
            anim.save(f'{self.plot_dir}/{self.algo_out_name}_{self.imp_out_name}_{self.fold}_batch{batch}_agent{agent}.gif', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default='zara1', help='select which fold')
    parser.add_argument('--algo', default='sgnet', help='select which fold')
    parser.add_argument('--imp', default='naomi', choices=['linear', 'smooth', 'naomi'], help='select which fold')
    parser.add_argument('--out', default='./out/', help='out folder')
    parser.add_argument('--cutoffs', action='store_true', help='get cutoffs only')
    args = parser.parse_args()

    output_dir = args.out
    fold = args.fold
    algo = args.algo
    imp = args.imp
    get_cutoffs = args.cutoffs
    algo_map = {'sgnet': 'SGNet', 'ego_vrnn': 'VRNN', 'ego_avrnn': 'A-VRNN'}
    fold_map = {'eth': 'ETH', 'hotel': 'Hotel', 'univ': 'Univ', 'zara1': 'Zara1', 'zara2': 'Zara2'}
    if not get_cutoffs:
        algo_name = algo_map[algo]
        fold_name = fold_map[fold]
        run_base = f'det_train_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
        if imp == 'linear':
            run_imp = f'det_train_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
            run = f'det_train_corr_e2e_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
        else:
            run_imp = f'det_train_{imp}_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
            run = f'det_train_corr_e2e_{imp}_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'

        # Good ones: sgnet, zara2, batch10 agent133
        hist_len = 8
        corr = True
        smooth = False
        vis = Visualizer(output_dir=output_dir, corr = corr, smooth = smooth, hist_len = hist_len,
                        algo_name=algo_name, algo_out_name = algo, fold_name=fold_name, imp_out_name = imp)
        vis.load_data(fold=fold, run=run, epoch=None, run_base=run_base, run_imp = run_imp)
    else:
        algo_name = algo_map[algo]
        fold_name = fold_map[fold]
        run_base = f'det_train_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
        all_cutoffs = set()
        for imp in ['linear', 'smooth', 'naomi']:
            if imp == 'linear':
                run_imp = f'det_train_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
                run = f'det_train_corr_e2e_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
            else:
                run_imp = f'det_train_{imp}_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
                run = f'det_train_corr_e2e_{imp}_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'

            # Good ones: sgnet, zara2, batch10 agent133
            hist_len = 8
            corr = True
            smooth = False
            vis = Visualizer(output_dir=output_dir, corr = corr, smooth = smooth, hist_len = hist_len,
                            algo_name=algo_name, algo_out_name = algo, fold_name=fold_name, imp_out_name = imp)
            cutoffs = vis.load_data(fold=fold, run=run, epoch=None, run_base=run_base, run_imp = run_imp, get_cutoffs=True)
            if not len(all_cutoffs):
                all_cutoffs = set(cutoffs)
            else:
                all_cutoffs = all_cutoffs.intersection(set(cutoffs))
        for imp in ['linear', 'smooth', 'naomi']:
            if imp == 'linear':
                run_imp = f'det_train_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
                run = f'det_train_corr_e2e_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
            else:
                run_imp = f'det_train_{imp}_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'
                run = f'det_train_corr_e2e_{imp}_{fold}_{algo}_rel_2d_hl-8_hs-10_fl-12_fs-10'

            # Good ones: sgnet, zara2, batch10 agent133
            hist_len = 8
            corr = True
            smooth = False
            vis = Visualizer(output_dir=output_dir, corr = corr, smooth = smooth, hist_len = hist_len,
                            algo_name=algo_name, algo_out_name = algo, fold_name=fold_name, imp_out_name = imp)
            vis.load_data(fold=fold, run=run, epoch=None, run_base=run_base, run_imp = run_imp, to_render=all_cutoffs)
