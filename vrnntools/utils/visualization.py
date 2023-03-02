# ------------------------------------------------------------------------------
# @file:    visualizaton.py
# @brief:   This file contains the implementation of visualization utils
# @author:  Ingrid Navarro
# @date:    Last modified on March 15th, 2022
# ------------------------------------------------------------------------------

import logging
from re import S
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import random
import torch

from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from natsort import natsorted

logger = logging.getLogger(__name__)

cmaps_list = ['Blues', 'Greys', 'Purples', 'Greens', 'Oranges', 'Reds']

# trajectory plots

def plot_trajectories(
    config, hist, fut, pred, seq_start_end, best_sample_idx, filename='val_'):
    """ Trajectory plot wrapper.
    Inputs:
    -------
    config[dict]: visualization configuration parameters
    hist[torch.tensor(hist_len, dim, num_batch)]: trajectories history
    fut[torch.tensor(fut_len, dim, num_batch)]: trajectories future
    pred[torch.tensor(num_samples, fut_len, dim, num_batch)]: trajectories predictions
    dataset_name[str]: name of trajectory dataset
    seq_start_end[int]: agents' sequences to plot
    filename[str]: filename
    """
    for i, (s, e) in enumerate(seq_start_end):
        idx = best_sample_idx[s:e]
        fn = filename+ f"_seq-{i}"
        h, f, p_multi = hist[:, :, s:e], fut[:, :, s:e], pred[:, :, :, s:e]
        
        # TODO: fix this
        n = e - s
        p = np.empty_like(f)
        for j in range(n):
            p[:, :, j] = p_multi[best_sample_idx[j], :, :, j]
            
        plot_2d_trajectories(
            config=config, hist=h, fut=f, pred=p, max_agents=n, filename=fn)
        
        plot_2d_trajectories_multimodal(
            config=config, hist=h, fut=f, pred=p_multi, best_idx=idx, 
            max_agents=n, filename=fn)
        
        if config.animation:
            animate_2d_trajectories(
                config=config, hist=h, fut=f, pred=p, max_agents=n, filename=fn)
    
    # TODO: add plot patterns
    
    # if "bsk" in dataset_name:
    #     plot_trajectories_bsk(
    #         config=config, hist=hist, fut=fut, pred=pred, 
    #         max_agents=max_agents, filename=filename)
    # # traj-air plots
    # elif "days" in dataset_name:
    # elif "sdd" in dataset_name:
    #     plot2d_trajectories(
    #         config, hist, fut, pred, max_agents=max_agents, filename=filename)
    # else:
    #     logger.info(f"Dataset {dataset_name} is not supported!")

def plot_2d_trajectories(
    config, hist, fut, pred, max_agents = 100, filename='val_'):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    ax = plt.axes()
    if config.use_limits:
        ax = plt.axes(
            xlim=(config.x_lim[0], config.x_lim[1]),
            ylim=(config.y_lim[0], config.y_lim[1]))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    _, _, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    mhist, mfut, mpred, alpha = '--', '--', '-', 1.0
    lw = config.lw
    
    # plot agent trajectoreis (history and future)
    for agent in range(num_agents):
        # append legent to first agent
        if agent == 0:
            # airport landmark
            plt.plot(
                [0], [0.1], 's', color='C'+str(num_agents % 10), label="Airport", 
                markersize=8, alpha=0.7)
            
            # history start landmark
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='History Start')
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w' )
            
            # prediction start landmark
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Prediction Start')
            
            # goal landmark
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='Goal')
            
            # history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Ground Truth Trajectory')
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Prediction')
        else:
            # a circle will denote the start location along with agent number
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w')
            
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # a star will denote the goal 
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            
            # agent history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
    # add background
    if os.path.exists(config.background):
        background = plt.imread(config.background)
        plt.imshow(background, zorder=0,
            extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                    config.x_lim[0], config.y_lim[1]], alpha=0.8)
    
    lgnd = plt.legend(
        fontsize=6, loc="upper center", ncol=3, labelspacing = 1, handletextpad=0.3)

    out_file = os.path.join(config.plot_path, f"{filename}.png")
    logger.debug(f"Saving plot to {out_file}")
    plt.savefig(out_file, format=config.format, dpi=config.dpi, bbox_inches='tight')
    plt.close()


def plot_2d_trajectories_multimodal(
    config, hist, fut, pred, best_idx, max_agents = 100, filename='val_'):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    ax = plt.axes()
    if config.use_limits:
        ax = plt.axes(
            xlim=(config.x_lim[0], config.x_lim[1]),
            ylim=(config.y_lim[0], config.y_lim[1]))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    _, _, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    mhist, mfut, mpred, alpha = '--', '--', '-', 1.0
    lw = config.lw
    
    # plot agent trajectoreis (history and future)
    for agent in range(num_agents):
        alpha = 1.0
        # append legent to first agent
        if agent == 0:
            # airport landmark
            plt.plot(
                [0], [0.1], 's', color='C'+str(num_agents % 10), label="Airport", 
                markersize=8, alpha=0.7)
            
            # history start landmark
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='History Start')
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w' )
            
            # prediction start landmark
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Prediction Start')
            
            # goal landmark
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='Goal')
            
            # history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Ground Truth Trajectory')
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            for i in range(pred.shape[0]):
                if not i == best_idx[agent]:
                    alpha = 0.7
                    mpred = ':'
                    label = ''
                else:
                    alpha = 1.0
                    mpred = '-'
                    label = 'Prediction'
                px = np.append(hist[-1, 0, agent], pred[i, :, 0, agent])
                py = np.append(hist[-1, 1, agent], pred[i, :, 1, agent])
                plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                    markersize=1, alpha=alpha, label=label)

        else:
            # a circle will denote the start location along with agent number
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w')
            
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # a star will denote the goal 
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            
            # agent history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            for i in range(pred.shape[0]):
                if not i == best_idx[agent]:
                    alpha = 0.7
                    mpred = ':'
                else:
                    alpha = 1.0
                    mpred = '-'
                px = np.append(hist[-1, 0, agent], pred[i, :, 0, agent])
                py = np.append(hist[-1, 1, agent], pred[i, :, 1, agent])
                plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                    markersize=1, alpha=alpha)

    # add background
    if os.path.exists(config.background):
        background = plt.imread(config.background)
        plt.imshow(background, zorder=0,
            extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                    config.x_lim[0], config.y_lim[1]], alpha=0.8)
    
    lgnd = plt.legend(
        fontsize=6, loc="upper center", ncol=3, labelspacing = 1, handletextpad=0.3)
 
    out_file = os.path.join(config.plot_path, f"{filename}_multi.png")
    logger.debug(f"Saving plot to {out_file}")
    plt.savefig(out_file, format=config.format, dpi=config.dpi, bbox_inches='tight')
    plt.close()

def animate_2d_trajectories(
    config, hist, fut, pred, max_agents = 100, filename='val_'):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    ax = plt.axes(
        xlim=(config.x_lim[0], config.x_lim[1]),
        ylim=(config.y_lim[0], config.y_lim[1]))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    _, _, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    mhist, mfut, mpred, alpha = '--', '--', '-', 1.0
    lw = config.lw
    
    # plot agent trajectoreis (history and future)
    agent_trajs_x, agent_trajs_y = [], []
    for agent in range(num_agents):
        # append legent to first agent
        if agent == 0:
            # airport landmark
            plt.plot(
                [0], [0.1], 's', color='C'+str(num_agents % 10), label="Airport", 
                markersize=8, alpha=0.7)
            
            # history start landmark
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='History Start')
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w' )
            
            # prediction start landmark
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Prediction Start')
            
            # goal landmark
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='Goal')
            
            # history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Ground Truth Trajectory')
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            agent_trajs_x.append(px)
            agent_trajs_y.append(py)
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=0.2, label='Prediction')
        else:
            # a circle will denote the start location along with agent number
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w')
            
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # a star will denote the goal 
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            
            # agent history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            agent_trajs_x.append(px)
            agent_trajs_y.append(py)
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=0.2)
    
    plt.legend(fontsize=6, loc="upper center", ncol=3, labelspacing = 1, 
        handletextpad=0.3)
    
    # add background
    if os.path.exists(config.background):
        background = plt.imread(config.background)
        plt.imshow(background, zorder=0,
            extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                    config.x_lim[0], config.y_lim[1]], alpha=0.8)
    
    # animate predictions 
    agents_x = [[] for _ in range(num_agents)]
    agents_y = [[] for _ in range(num_agents)]
        
    def animate(i):
        for agent in range(num_agents):
            agents_x[agent].append(agent_trajs_x[agent][i])
            agents_y[agent].append(agent_trajs_y[agent][i])
            plt.plot(
                agents_x[agent], agents_y[agent], mpred, color='C'+str(agent % 10), 
                linewidth=lw, markersize=1)
    
    anim = animation.FuncAnimation(
        fig, animate, frames=agent_trajs_y[0].shape[0], interval=300)
    
    out_file = os.path.join(config.video_path, f"{filename}.gif")
    logger.debug(f"Saving animation to {out_file}")
    anim.save(out_file, dpi=config.dpi)
    
# ------------------------------------------------------------------------------
# Everything below is not being used
# ------------------------------------------------------------------------------

def heat2rgb(t):
    def clamp(v):
        return max(0, min(v, 1))
    r = clamp(1.5 - abs(2.0*t-1.0))
    g = clamp(1.5 - abs(2.0*t))
    b = clamp(1.5 - abs(2.0*t+1.0))
    return (r, g, b)


def lighten_color(color, amount=0.3):
    import matplotlib.colors as mc
    import colorsys

    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_views(
    ax, angles, elevation=None, width=8, height=8, prefix='tmprot_', **kwargs
):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created. 

    Returns: the list of files created (for later removal)
    """
    files = []
    ax.figure.set_size_inches(width, height)
    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = '%s%03d.jpeg' % (prefix, i)
        ax.figure.savefig(fname, bbox_inches='tight', dpi=300)
        files.append(fname)

    return files

def traj_animate(num, data_pred, data_patt, line, dots, patts):
    line.set_data(data_pred[0:2, :num])    
    dots.set_data(data_pred[0:2, :num])
    patts[num].set_data(data_patt[0:2, :num])
    return line

def make_gif(
    files, output, delay=100, repeat=True, **kwargs
):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              % (delay, loop, " ".join(files), output))


def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation from a 3D plot on a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """
    files = make_views(ax, angles, **kwargs)
    make_gif(files, output, **kwargs)
    for f in files:
        os.remove(f)

def make_video_from_files(in_path, pattern, out_path, filename):
    list_images = os.listdir(in_path)
    weights_images = [img for img in list_images if ".gif" not in img]
    weights_images = [img for img in list_images if str(pattern) in img]
    weights_images = natsorted(weights_images)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    create_video(
        weights_images, base_path=in_path, out_path=out_path, filename=filename
    )

# ------------------------------------------------------------------------------
# TRAJECTORIES


def plot_trajectories_bsk(config, hist, fut, pred, max_agents = 100, filename='val_'):
    """ Plots full 2d trajectories, i.e. hist + ground truth / predictions.
    Credit to: https://github.com/linouk23/NBA-Player-Movements
    Args
        config: visualization configuration
        hist: trajectories' histories
        fut: trajectories' ground truth future
        pred: trajectories' future predictions
        filename: file name
    """
    # set the limits 
    ax = plt.axes(
        xlim=(config.x_lim[0], config.x_lim[1]),
        ylim=(config.y_lim[0], config.y_lim[1]))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    hist_len, num_dim, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    agents = []
    if num_agents > 5:
        agents = random.sample(range(num_agents), 10) # [0, 5]

    # plot agent trajectoreis (history and future)
    for agent in range(num_agents):
        alpha = 0.3
        mhist, mfut, alpha = '-', '-', 0.3
        if agent in agents:
            mfut, alpha = ':', 0.6
        
        # a circle will denote the start location along with agent number
        x, y = hist[0, 0, agent], hist[0, 1, agent]
        plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), alpha=alpha)
        ax.annotate(str(agent+1), xy=(x, y), fontsize=6)
        
        plt.plot(hist[:, 0, agent], hist[:, 1, agent], mhist, 
            color='C'+str(agent % 10), linewidth=1, markersize=1, alpha=alpha)
        
        plt.plot(
            np.append(hist[-1, 0, agent], fut[:, 0, agent]),
            np.append(hist[-1, 1, agent], fut[:, 1, agent]),
            mfut, color='C'+str(agent % 10), linewidth=1, markersize=1, 
            alpha=alpha)
        
        if agent in agents:
            team = 'atk' if agent < 5 else 'def'
            plt.plot(
                np.append(hist[-1, 0, agent], pred[:, 0, agent]),
                np.append(hist[-1, 1, agent], pred[:, 1, agent]),
                '--', color='C'+str(agent % 10), linewidth=1, markersize=1, 
                label=team, alpha=alpha)
        
        # a star will denote the goal 
        plt.plot(fut[-1, 0, agent], fut[-1, 1, agent], '*', markersize=10, 
            color='C'+str(agent % 10), alpha=alpha)
    
    # add basketball court
    court = plt.imread(config.background)
    plt.imshow(court, zorder=0,
        extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                config.x_lim[0], config.y_lim[1]], alpha=0.5)
    
    plt.plot([0], [0], ':', color='k', label="Future", markersize=1)
    plt.plot([0], [0], '--', color='k', label="Prediction", markersize=2)
    plt.plot([0], [0], '-', color='k', label="History", markersize=2)
    plt.plot([0], [0], '8', color='k', label="Start", markersize=2)
    plt.plot([0], [0], '*', color='k', label="Goal", markersize=2)
    plt.legend()

    # animate predictions 
    agent_one_x, agent_one_y = [], []
    agent_one_traj_x = np.append(hist[-1, 0, agents[0]], pred[:, 0, agents[0]])
    agent_one_traj_y = np.append(hist[-1, 1, agents[0]], pred[:, 1, agents[0]])
    agent_two_x, agent_two_y = [], []
    agent_two_traj_x = np.append(hist[-1, 0, agents[1]], pred[:, 0, agents[1]])
    agent_two_traj_y = np.append(hist[-1, 1, agents[1]], pred[:, 1, agents[1]])
    
    def animate(i):
        agent_one_x.append(agent_one_traj_x[i])
        agent_one_y.append(agent_one_traj_y[i])
        agent_two_x.append(agent_two_traj_x[i])
        agent_two_y.append(agent_two_traj_y[i])
    
        plt.plot(agent_one_x, agent_one_y, '--', color='C'+str(agents[0] % 10), 
            linewidth=2, markersize=1)
        plt.plot(agent_two_x, agent_two_y, '--', color='C'+str(agents[1] % 10), 
            linewidth=2, markersize=1)
    
    anim = animation.FuncAnimation(
        fig, animate, frames=agent_one_traj_x.shape[0], interval=300)
    out_file = os.path.join(config.path, f"{filename}.gif")
    logger.debug(f"Plotting val batch to {out_file}")
    anim.save(out_file, dpi=config.dpi)
    
    logger.debug(f"Plotting val batch to {out_file}")
    out_file = os.path.join(config.path, f"{filename}.png")
    plt.savefig(out_file, format=config.format, dpi=config.dpi, bbox_inches='tight')
    plt.close()
   

def plot_agent_patterns(config, hist, fut, pred, patterns, filename='val_'):
    """ Plots full trajectories with patterns i.e, hist + ground truth / predictions.
    Args
        config: visualization configuration
        hist: trajectories' histories
        fut: trajectories' ground truth future
        pred: trajectories' future predictions
        filename: file name
    """
    hist_len, num_dim, num_agents = hist.shape  # n_channel is 2
    agent = random.randint(0, num_agents-1)
    
    if torch.is_tensor(hist):
        hist = hist.numpy()

    if torch.is_tensor(fut):
        fut = fut.numpy()

    fig = plt.figure()
    
    # plt.ion() # plt.show()
    plt.clf()
    # if config.use_limits:
    #     plt.xlim(config.x_lim)
    #     plt.xlabel(config.x_label)
    #     plt.ylim(config.y_lim)
    #     plt.ylabel(config.y_label)
    
    plt.plot(
        hist[:, 0, agent], hist[:, 1, agent], '.-', 
        color='C'+str(agent % 10), linewidth=1, markersize=1, label='History')
    plt.plot(
        hist[-1, 0, agent], hist[-1, 1, agent], '*', 
        color='C'+str(agent % 10), linewidth=1, markersize=3)
    
    plt.plot(
        np.append(hist[-1, 0, agent], fut[:, 0, agent]),
        np.append(hist[-1, 1, agent], fut[:, 1, agent]),
        ':.', color='C'+str(agent % 10), linewidth=1, markersize=1, label='Future')
     
    x_pred = pred[:, 0, agent]
    y_pred = pred[:, 1, agent]
    
    x_patterns = patterns[:, agent, :, 0]
    y_patterns = patterns[:, agent, :, 1]
    
    data_pred = np.array([x_pred, y_pred])
    data_patt = np.array([x_patterns, y_patterns])
    num_points = fut.shape[0]
    
    dots = plt.plot(
        data_pred[0], data_pred[1], lw=1, markersize=2, color='C'+str(agent % 10), 
        marker='o')[0] 
    line = plt.plot(data_pred[0], data_pred[1], lw=1, c='C'+str(agent % 10))[0] 
    
    patts = []
    for t in range(num_points):
        patts.append(
            plt.plot(
                data_patt[0, t], data_patt[1, t], '.-', color='C'+str(agent % 10), 
                linewidth=1, markersize=1, alpha=0.1)[0])
        
    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, traj_animate, frames=num_points, 
        fargs=(data_pred, data_patt, line, dots, patts), 
        interval=500, blit=False)
    
    out_file = os.path.join(config.path, f"{filename}.gif")
    logger.debug(f"Plotting val batch to {out_file}")
    line_ani.save(out_file)
    plt.close()

def plot2d_trajectories(config, hist, fut=None, pred=None, max_agents = 100, filename="0"):
    """ Plots full 2d trajectories, i.e. hist + ground truth / predictions.
    Args
        config: visualization configuration
        hist: trajectories' histories
        fut: trajectories' ground truth future
        pred: trajectories' future predictions
        filename: file name
    """
    hist_len, num_dim, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    if torch.is_tensor(hist):
        hist = hist.numpy()

    if torch.is_tensor(fut):
        fut = fut.numpy()

    # plt.ion() # plt.show()
    plt.clf()
    if config.use_limits:
        plt.xlim(config.x_lim)
        plt.xlabel(config.x_label)
        plt.ylim(config.y_lim)
        plt.ylabel(config.y_label)
        
    for agent in range(num_agents):
        plt.plot(
            hist[:, 0, agent], hist[:, 1, agent], '.-', 
            color='C'+str(agent % 10), linewidth=1, markersize=1
        )
        plt.plot(
            hist[-1, 0, agent], hist[-1, 1, agent], '*', 
            color='C'+str(agent % 10), linewidth=1, markersize=3
        )

        if fut is not None:
            # plt.plot(fut_traj_list[i,0,:],fut_traj_list[i,1,:],':.',color='C'+str(i%10))
            plt.plot(
                np.append(hist[-1, 0, agent], fut[:, 0, agent]),
                np.append(hist[-1, 1, agent], fut[:, 1, agent]),
                ':.', color='C'+str(agent % 10), linewidth=1, markersize=1
            )
        if pred is not None:
            plt.plot(
                np.append(hist[-1, 0, agent], pred[:, 0, agent]),
                np.append(hist[-1, 1, agent], pred[:, 1, agent]),
                marker='<', color='C'+str(agent % 10), linewidth=1, markersize=1
            )

    plt.plot([0], [0], ':.', color='k', label="Future", markersize=2)
    plt.plot([0], [0], marker='<', color='k', label="Prediction", markersize=2)
    plt.plot([0], [0], '-o', color='k', label="History", markersize=2)
    plt.legend()
    # plt.axis("equal")

    out_file = os.path.join(config.path, f"{filename}.{config.format}")
    logger.debug(f"Plotting val batch to {out_file}")
    plt.savefig(out_file, format=config.format,
                dpi=config.dpi, bbox_inches='tight')
    plt.close()
    
def plot3d_trajectories(config, hist, fut=None, pred=None, max_agents = 100, filename="0"):
    """ Plots full 2d trajectories, i.e. hist + ground truth / predictions.
    Args
        config: visualization configuration
        hist: trajectories' histories
        fut: trajectories' ground truth future
        pred: trajectories' future predictions
        filename: file name
    """
    hist_len, num_dim, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    if torch.is_tensor(hist):
        hist = hist.numpy()

    if torch.is_tensor(fut):
        fut = fut.numpy()

    # plt.ion() # plt.show()
    plt.clf()
    if config.use_limits:
        plt.xlim(config.x_lim)
        plt.xlabel(config.x_label)
        plt.ylim(config.y_lim)
        plt.ylabel(config.y_label)
    
    # max height (z) value
    max_H = np.max(hist[:, 2, :num_agents])
    max_F = np.finfo(dtype=np.float).min
    max_P = np.finfo(dtype=np.float).min 
    if fut is not None:    
        max_F = np.max(fut[:, 2, :num_agents])
    if pred is not None:
        max_P = np.max(pred[:, 2, :num_agents])
        
    max_z = max(max_H, max(max_F, max_P))
    
    for agent in range(num_agents):
        Hx, Hy, Hz = hist[:, 0, agent], hist[:, 1, agent], hist[:, 2, agent]
        cmap = cm.get_cmap(cmaps_list[agent % 6])
        
        text = "A:{} H:{:.2f}".format(agent, max_H)
        
        for i in range(Hz.shape[0]):
            plt.plot(Hx, Hy, color=cmap(Hz[i] / max_z), linewidth=1, markersize=2)
        plt.plot(Hx[-1], Hy[-1], '*', color=cmap(Hz[-1] / max_z), linewidth=1, markersize=3)
        
        if fut is not None:
            text += " F: {:.2f}".format(max_F)
            F = np.concatenate((hist[-1, :, agent].reshape(1, -1), fut[:, :, agent]), axis=0)
            Fx, Fy, Fz = F[:, 0], F[:, 1], F[:, 2]
            
            for i in range(Fz.shape[0]):
                plt.plot(Fx, Fy, ':.', color=cmap(Fz[i]  / max_z), linewidth=1, markersize=1)
        
        if pred is not None:
            text += " P: {:.2f}".format(max_P)
            P = np.concatenate((hist[-1, :, agent].reshape(1, -1), pred[:, :, agent]), axis=0)
            Px, Py, Pz = P[:, 0], P[:, 1], P[:, 2]
            
            for i in range(Pz.shape[0]):
                plt.plot(Px, Py, marker='<', color=cmap(Pz[i]  / max_z), linewidth=1, markersize=1)
        
        if config.add_text:
            font = {'family': 'serif', 'color': cmap(1.0), 'size': 7}
            plt.text(x=config.x_lim[0], y=config.y_lim[0] + 0.5 * (agent+1), s=text, fontdict=font)
        
    plt.plot([0], [0], ':.', color='k', label="Future", markersize=2)
    plt.plot([0], [0], marker='<', color='k', label="Prediction", markersize=2)
    plt.plot([0], [0], '-o', color='k', label="History", markersize=2)
    plt.legend()
    # plt.axis("equal")

    ofile = os.path.join(config.path, f"{filename}.{config.format}")
    logger.debug(f"Plotting val batch to {ofile}")
    plt.savefig(ofile, format=config.format, dpi=config.dpi, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# PATTERNS

def plot_patterns(config, patterns, W=None, filename="000", fsize=8):
    """ Plots learned patterns 
    Args:
        patterns: first layer weights
        config: visualization configuration
        filename
    """
    _, dim, _ = patterns.shape

    if dim == 2:
        # TODO: add trajectory
        plot2d_patterns(config, patterns, W, filename, fsize)
    elif dim == 3:
        # TODO: add trajectory
        plot3d_patterns_2d(config, patterns, W, filename, fsize)
        filename = filename.replace("weights", "cm")
        plot3d_patterns_cm(config, patterns, W, filename, fsize)
    else:
        raise ValueError(f"Dim: {dim} not supported!")

def plot2d_patterns(
    config, patterns, W=None, filename="weights", fsize=6, alpha=None
):
    """ Plots learned 2d patterns 
    Args:
        patterns: first layer weights
        config: visualization configuration
        filename
    """
    plt.clf()
    fig = plt.figure(figsize=[fsize, fsize])    
    plt.plot(0, 0, "^", zorder=10)
    if torch.is_tensor(patterns):
        patterns = patterns.numpy().copy()

    plt.plot(
        [np.min(patterns[:, 1, :]), np.max(patterns[:, 1, :])],
        [np.min(patterns[:, 0, :]), np.max(patterns[:, 0, :])],
        'w.', zorder=10)

    if W is None:
        # np.arange(len(P))/len(P)*2-1# W = np.random.random(len(P))
        W = -np.ones(len(patterns))
        
    color = tuple(config.color) if config.color else None    
    for (p, w) in zip(patterns, W):
        # dp = p[:, 1] - p[:, 0]
        dp = p[:, 1] - p[:, 0]
        plt.arrow(
            p[0, 0], p[1, 0], dp[0], dp[1], length_includes_head=True,
            color=heat2rgb(w) if color is None else color,
            alpha=1.0 if alpha is None else alpha, 
            width=.01, zorder=10
        )
        
    # TODO: plot with trajectory
    # if targ_traj is not None:
    #     if torch.is_tensor(targ_traj):
    #         targ_traj = targ_traj.numpy()
    #     plt.plot(targ_traj[1, :], targ_traj[0, :], '^-', color='C0')

    # if Traj is not None:
    #     if torch.is_tensor(Traj):
    #         Traj = Traj.numpy()

    #     Traj = Traj.reshape(-1, 2, fsize)
    #     for i, traj in enumerate(Traj):
    #         plt.plot(traj[1, :], traj[0, :], '.-', color='C'+str((i+1) % 10))
    #         plt.plot(traj[1, -1], traj[0, -1], 'o', color='C'+str((i+1) % 10))
    
    plt.axis("equal")
    if config.airport and os.path.exists(config.airport):
        im = plt.imread(config.airport)
        plt.imshow(
            im, extent=[-6.0, 6.0, -6.0, 6.0], alpha=0.7, zorder=0
        )
        
    out_file = os.path.join(config.path, filename+".png")
    logger.debug(f"Plotting weights to {out_file}")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    
def plot3d_patterns_cm(
    config, patterns, W=None, filename="weights", fsize=8, color=None, alpha=None
):
    """ Plots learned 3d patterns with colormap
    Args:
        patterns: first layer weights
        config: visualization configuration
        filename
    """
    plt.clf()
    fig = plt.figure(figsize=[fsize, fsize])
    plt.plot(0, 0, "^")
    if torch.is_tensor(patterns):
        patterns = patterns.numpy().copy()

    plt.plot([np.min(patterns[:, 1, :]), np.max(patterns[:, 1, :])],
             [np.min(patterns[:, 0, :]), np.max(patterns[:, 0, :])], 'w.')

    if W is None:
        # np.arange(len(P))/len(P)*2-1# W = np.random.random(len(P))
        W = -np.ones(len(patterns))
    
    max_dpz = np.max(patterns[:, 2, 1] - patterns[:, 2, 0])
    min_dpz = np.min(patterns[:, 2, 1] - patterns[:, 2, 0])
    
    for (p, w) in zip(patterns, W):
        # dp = p[:, 1] - p[:, 0]
        dp = p[:, 1] - p[:, 0]
        plt.arrow(
            p[0, 0], p[1, 0], dp[0], dp[1], length_includes_head=True,
            color=cm.hot(dp[2] / max_dpz),
            alpha=1.0 if alpha is None else alpha, 
            width=.01
        )
        
    # TODO: plot with trajectory
    # if targ_traj is not None:
    #     if torch.is_tensor(targ_traj):
    #         targ_traj = targ_traj.numpy()
    #     plt.plot(targ_traj[1, :], targ_traj[0, :], '^-', color='C0')

    # if Traj is not None:
    #     if torch.is_tensor(Traj):
    #         Traj = Traj.numpy()

    #     Traj = Traj.reshape(-1, 2, fsize)
    #     for i, traj in enumerate(Traj):
    #         plt.plot(traj[1, :], traj[0, :], '.-', color='C'+str((i+1) % 10))
    #         plt.plot(traj[1, -1], traj[0, -1], 'o', color='C'+str((i+1) % 10))
    
    if config.add_text:
        text = "height max: {:.3f} min: {:.3f}".format(max_dpz, min_dpz)
        font = {'family': 'serif', 'color': 'black', 'size': 7}
        plt.text(x=config.x_lim[0], y=config.y_lim[0] - 1.5, s=text, fontdict=font)

    plt.axis("equal")
    sm = cm.ScalarMappable(colors.Normalize(min_dpz, max_dpz), cmap=cm.hot)
    plt.colorbar(sm)
    out_file = os.path.join(config.path, filename+".png")
    logger.debug(f"Plotting weights to {out_file}")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def plot3d_patterns_2d(
    config, patterns, W=None, filename="weights", fsize=8, color=None, alpha=None
):
    """ Plots learned 3d patterns in 2d planes (i.e., xy, xz, yz)
    Args:
        patterns: first layer weights
        config: visualization configuration
        filename
    """
    plt.clf()
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(3 * fsize, fsize))

    if torch.is_tensor(patterns):
        patterns = patterns.numpy().copy()

    ax_xy.plot(0, 0, "^")
    ax_xy.set_title("x-y")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    # ax_xy.plot(
    #     [np.min(patterns[:, 1, :]), np.max(patterns[:, 1, :])],
    #     [np.min(patterns[:, 0, :]), np.max(patterns[:, 0, :])], 'w.'
    # )

    ax_xz.plot(0, 0, "^")
    ax_xz.set_title(f"x-z")
    ax_xz.set_xlabel(f"x")
    ax_xz.set_ylabel(f"z")
    # ax_xz.plot(
    #     [np.min(patterns[:, 1, :]), np.max(patterns[:, 1, :])],
    #     [np.min(patterns[:, 0, :]), np.max(patterns[:, 0, :])], 'w.'
    # )
    
    ax_yz.plot(0, 0, "^")
    ax_yz.set_title("y-z")
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    # ax_yz.plot(
    #     [np.min(patterns[:, 1, :]), np.max(patterns[:, 1, :])],
    #     [np.min(patterns[:, 0, :]), np.max(patterns[:, 0, :])], 'w.'
    # )

    if W is None:
        # np.arange(len(P))/len(P)*2-1# W = np.random.random(len(P))
        W = -np.ones(len(patterns))

    # pattern shape is (n_pattern, ker_size, n_channel)
    for (p, w) in zip(patterns, W):
        # p (2, 3)
        # dp = p[:, 1] - p[:, 0]
        dp = p[:, 1] - p[:, 0]
        ax_xy.arrow(
            p[0, 0], p[1, 0], dp[0], dp[1], 
            length_includes_head=True,
            color=heat2rgb(w) if color is None else color,
            alpha=1.0 if alpha is None else alpha, width=.01
        )
        ax_xz.arrow(
            p[0, 0], p[2, 0], dp[0], dp[2], 
            length_includes_head=True,
            color=heat2rgb(w) if color is None else color,
            alpha=1.0 if alpha is None else alpha, width=.01
        )
        ax_yz.arrow(
            p[1, 0], p[2, 0], dp[1], dp[2], 
            length_includes_head=True,
            color=heat2rgb(w) if color is None else color,
            alpha=1.0 if alpha is None else alpha, width=.01
        )
    
    plt.axis("equal")
    out_file = os.path.join(config.path, filename+".png")
    logger.debug(f"Plotting weights to {out_file}")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot3d_patterns_3d(config, patterns, W=None, filename="weights"):
    """ Plots learned 3d patterns in 3d 
    Args:
        patterns: first layer weights
        config: visualization configuration
        filename
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    mn = 2 * min(np.min(patterns[:, :, 0]), 
                 min(np.min(patterns[:, :, 1]), np.min(patterns[:, :, 2])))
    mx = 2 * max(np.max(patterns[:, :, 0]), 
                 max(np.max(patterns[:, :, 1]), np.max(patterns[:, :, 2])))

    ax.axes.set_xlim3d(left=mn, right=mx)
    ax.axes.set_ylim3d(bottom=mn, top=mx)
    ax.axes.set_zlim3d(bottom=mn, top=mx)

    for (p, w) in zip(patterns, W):
        x1, x2 = p[1, 0], p[0, 0]
        y1, y2 = p[1, 1], p[0, 1]
        z1, z2 = p[1, 2], p[0, 2]
        ax.quiver(x1, y1, z1, x2, y2, z2)
        # color=heat2rgb(w) if _c is None else _c,
        # alpha=1.0 if _a is None else _a, width=.01

    angles = np.linspace(0, 360, 21)[:-1]
    out_file = os.path.join(config.path, filename+".gif")
    rotanimate(ax, angles, out_file, delay=30)
    logger.debug(f"Plotting val batch to {out_file}")
    plt.close()

    # TODO: fix this
    # if targ_traj is not None:
    #     if torch.is_tensor(targ_traj):
    #         targ_traj = targ_traj.numpy()
    #     plt.plot(targ_traj[1,:], targ_traj[0,:], '^-', color='C0')

    # if Traj is not None:
    #     if torch.is_tensor(Traj):
    #         Traj = Traj.numpy()

    #     Traj = Traj.reshape(-1,2, fsize)
    #     for i,traj in enumerate(Traj):
    #         plt.plot(traj[1,:], traj[0,:], '.-', color='C'+str((i+1)%10))
    #         plt.plot(traj[1,-1], traj[0,-1], 'o', color='C'+str((i+1)%10))
