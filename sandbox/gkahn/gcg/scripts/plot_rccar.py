import os
import numpy as np
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm

from load_experiment import Experiment
from sandbox.gkahn.gcg.utils.utils import DataAverageInterpolation

EXP_FOLDER = '/home/gkahn/code/rllab/data/local/sim-rccar'
SAVE_FOLDER = '/home/gkahn/code/rllab/data/local/sim-rccar/final_plots'
DT = 0.25

########################
### Load experiments ###
########################

def load_experiments(fnames, create_new_envs=False, load_train_rollouts=False, load_eval_rollouts=False):
    exps = []
    for fname in fnames:
        try:
            exps.append(Experiment(os.path.join(EXP_FOLDER, fname),
                                   clear_obs=True,
                                   create_new_envs=create_new_envs,
                                   load_train_rollouts=load_train_rollouts,
                                   load_eval_rollouts=load_eval_rollouts))
            print(fname)
        except:
            pass

    return exps

############
### Misc ###
############

def moving_avg_std(idxs, data, window):
    avg_idxs, means, stds = [], [], []
    for i in range(window, len(data)):
        avg_idxs.append(np.mean(idxs[i - window:i]))
        means.append(np.mean(data[i - window:i]))
        stds.append(np.std(data[i - window:i]))
    return avg_idxs, np.asarray(means), np.asarray(stds)

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20, success_cumreward=None, ylim=(10, 60),
                   plot_indiv=True, convert_to_time=False, xmax=None):
    data_interp = DataAverageInterpolation()
    if 'type' not in analyze_group[0].params['alg'] or analyze_group[0].params['alg']['type'] == 'interleave':
        min_step = max_step = None
        for i, analyze in enumerate(analyze_group):
            try:
                steps = np.asarray(analyze.progress['Step'], dtype=np.float32)
                values = np.asarray(analyze.progress['EvalCumRewardMean'])
                num_episodes = int(np.median(np.asarray(analyze.progress['EvalNumEpisodes'])))
                steps, values, _ = moving_avg_std(steps, values, window=window//num_episodes)

                # steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
                # values = np.array([np.sum(r['rewards']) for r in itertools.chain(*analyze.eval_rollouts_itrs)])
                #
                # steps, values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))
                # steps, values = zip(*[(s, v) for s, v in zip(steps, values) if np.isfinite(v)])
                #
                # steps, values, _ = moving_avg_std(steps, values, window=window)

                if plot_indiv:
                    assert(not convert_to_time)
                    ax.plot(steps, values, color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

                data_interp.add_data(steps, values)
            except:
                continue

            if min_step is None:
                min_step = steps[0]
            if max_step is None:
                max_step = steps[-1]
            min_step = max(min_step, steps[0])
            max_step = min(max_step, steps[-1])

        if len(data_interp.xs) == 0:
            return

        steps = np.r_[min_step:max_step:50.][1:-1]
        values_mean, values_std = data_interp.eval(steps)
        # steps -= min_step
    else:
        raise NotImplementedError

    if convert_to_time:
        steps = (DT / 3600.) * steps

    ax.plot(steps, values_mean, color=color, label=label)
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color=color, alpha=0.4)

    xmax = xmax if xmax is not None else max(steps)
    ax.set_xticks(np.arange(0, xmax, 1e4 if not convert_to_time else 1), minor=True)
    ax.set_xticks(np.arange(0, xmax, 3e4 if not convert_to_time else xmax // 4), minor=False)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    if not convert_to_time:
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)
    if ylim is not None:
        ax.set_ylim(ylim)

    if success_cumreward is not None:
        if not hasattr(success_cumreward, '__iter__'):
            success_cumreward = [success_cumreward]

        for i, sc in enumerate(success_cumreward):
            if values_mean.max() >= sc:
                thresh_step = steps[(values_mean >= sc).argmax()]
                # color = cm.viridis(i / float(len(success_cumreward)))
                color = ['b', 'm', 'c', 'r'][i]
                ax.vlines(thresh_step, *ax.get_ylim(), color=color, linestyle='--')

############################
### Specific experiments ###
############################

def plot_cluttered_hallway_ours():
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    FILE_NAME = 'cluttered_hallway_ours'
    all_exps = []
    all_exps.append(load_experiments(['ours_{0}'.format(i) for i in range(0, 3)]))
    all_exps.append(load_experiments(['ours_target_{0}'.format(i) for i in range(0, 3)]))
    titles = ['Ours', 'Ours Target']
    colors = ['g', 'b']

    f_cumreward, axes_cumreward = plt.subplots(1, 1, figsize=(6, 3), sharey=False, sharex=True)
    if not hasattr(axes_cumreward, '__len__'):
        axes_cumreward = np.array([axes_cumreward])

    window = 32
    ylim = (0, 2100)
    xmax_timesteps = 8e5

    for exp, title, color in zip(all_exps, titles, colors):
        plot_cumreward(axes_cumreward[0], exp, window=window, ylim=ylim, plot_indiv=False, convert_to_time=True, label=title, color=color)

    # set same x-axis
    xmax = xmax_timesteps * (DT / 3600.)
    for ax in axes_cumreward:
        ax.set_ylim(ylim)
        ax.set_xlim((0, xmax))
        ax.yaxis.set_ticks(np.arange(0, 2100, 500))
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.yaxis.set_ticks_position('both')

        # leg = ax.legend(ncol=len(all_exps), loc='upper center', bbox_to_anchor=(0.5, 1.3), columnspacing=1., fontsize=13)
        leg = ax.legend(ncol=1, loc='upper center', bbox_to_anchor=(1.4, 1.), columnspacing=1., handlelength=1.)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)

    f_cumreward.text(0.5, -0.05, 'Time (hours)', ha='center', fontdict=font)

    ax = axes_cumreward[0]
    ax.set_yticklabels(['', '250', '500', '750', '1000', ''])
    ax.set_ylabel('Distance (m)', fontdict=font)

    # add y-axis on right side which is hallway lengths
    ax = axes_cumreward[-1]
    ax_twin = ax.twinx()
    ax_twin.set_xlim((ax.get_xlim()))
    ax_twin.set_ylim((ax.get_ylim()))
    ax_twin.set_yticks(ax.get_yticks())
    ax_twin.yaxis.tick_right()
    ax_twin.set_yticklabels(['', '3', '6', '9', '12', ''])
    ax_twin.set_ylabel('Hallway lengths', fontdict=font)
    ax_twin.yaxis.set_label_position("right")

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
    plt.close(f_cumreward)

plot_cluttered_hallway_ours()
