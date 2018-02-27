import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from gcg.data import mypickle

class BnnPlotter(object):

    def __init__(self, preds, labels, train_preds=None, dones=None, env_infos=None):
        """
        preds: num_replays x num_bnn_samples x action_len-1
        labels: num_replays x action_len
        :return:
        """
        self.preds = preds
        self.labels = labels
        self.dones = dones
        self.env_infos = env_infos
        self.num_replays = preds.shape[0]
        self.num_bnn_samples = preds.shape[1]
        self.num_time_steps_pred = preds.shape[2]
        self.num_time_steps_replay = labels.shape[1]
        self.horizon = labels.shape[1]
        self.train_preds = train_preds
        assert preds.shape[0] == labels.shape[0]

    def save_all_plots(self, dir, plot_types=None):
        import matplotlib.pyplot as plt

        if plot_types is None:
            plot_types = [
                'plot_hist_time_to_collision',
                'plot_percentile_thresh_cost',
                # 'plot_pred_timeseries'
            ]

        for plot_type in plot_types:
            plot_method = getattr(self, plot_type)
            fig = plot_method()
            file_path = os.path.join(dir, plot_type + '.png')
            fig.savefig(file_path, bbox_inches='tight')
            plt.close(fig)

    def plot_pred_timeseries(self, num_replays_to_plot=10):
        import matplotlib.pyplot as plt
        num_replays_to_plot = min(num_replays_to_plot, self.num_replays)
        fig, axs = plt.subplots(num_replays_to_plot, self.num_time_steps_pred,
                                sharex=True, sharey=True, tight_layout=False, figsize=(18, 10))
        for i_replay in range(num_replays_to_plot):
            i_sample = np.random.randint(self.num_replays)
            for i_time in range(self.num_time_steps_pred):
                ax = axs[i_replay, i_time]
                x = self.preds[i_sample, :, i_time]
                color = 'b' if self.labels[i_sample, i_time] == 0 else 'r'
                ax.hist(x, 20, range=(0.,1.), color=color)
                if i_replay == num_replays_to_plot - 1:
                    ax.set_xlabel("t = {}".format(i_time))
                if i_time == 0:
                    ax.set_ylabel("sample {}".format(i_replay+1))
        return fig

    def plot_hist_time_to_collision(self):
        truth = self.num_time_steps_replay - self.labels.sum(axis=1)  # num_replays
        pred = self.num_time_steps_pred - self.preds.mean(axis=1).sum(axis=1)  # num_replays
        pred_std = self.preds.std(axis=1).mean(axis=1)

        num_bins = 20
        counts, bin_edges = np.histogram(pred - truth, bins=num_bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_indices = np.clip(np.digitize(pred - truth, bin_edges) - 1, 0, num_bins - 1) # TODO might be off by one?

        counts_std = []
        for i in range(num_bins):
            bin_indices_i = [bin_indices == i][0]
            if max(bin_indices_i) == False:
                counts_std.append(0)
            else:
                counts_std.append(pred_std[bin_indices_i].mean())
        counts_std = np.asarray(counts_std)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        y = counts / counts.sum()
        width = 0.9 * ((bin_edges.max() - bin_edges.min()) / num_bins)
        axes[0].bar(bin_centers, counts, width=width, color='b')
        axes[1].bar(bin_centers, counts_std, width=0.25 * width, color='r')

        for ax in axes:
            ax.set_xlabel("time steps (relative)")
        axes[0].set_title("predicted minus truth of time step collided (mean)")
        axes[1].set_title("predicted minus truth of time step collided (std)")

        fig.tight_layout()
        return fig

    def plot_percentile_thresh_cost(self, alpha_coll=100):
        assert (self.train_preds is not None)

        truth = self.num_time_steps_replay - self.labels.sum(axis=1)  # num_replays
        pred = self.num_time_steps_pred - self.preds.mean(axis=1).sum(axis=1)  # num_replays

        mean = pred - truth
        std = self.preds.std(axis=1).mean(axis=1)

        train_preds_std_avg = self.train_preds.std(axis=1).mean(axis=1).ravel()
        pct_threshes = np.linspace(0., 1., 1000)

        costs = []
        for pct_thresh in pct_threshes:

            ### calculate std threshold from train_labels
            std_thresh = np.percentile(train_preds_std_avg, pct_thresh)

            running_cost = 0
            for mean_i, std_i in zip(mean, std):
                ### if above std_thresh
                if std_i >= std_thresh:
                    ### cost is horizon
                    running_cost += self.horizon
                ### else below std_thresh
                else:
                    ### mean > 0
                    if mean_i > 0:
                        # then cost is alpha_coll
                        running_cost += alpha_coll
                    ### else mean < 0
                    else:
                        # then cost is abs(mean)
                        running_cost += abs(mean_i)

            costs.append(running_cost)


        f, ax = plt.subplots(1, 1)
        ax.plot(pct_threshes, costs, color='r')
        ax.set_xlabel('pct threshold')
        ax.set_ylabel('cost')
        ax.set_title('cost vs pct threshold')
        f.tight_layout()
        return f

    def plot_online_switching(self, H_avert=4, H_takeover=4, H_signal=4, save_dir=None, mean_thresh=0, std_thresh=None):
        import matplotlib.pyplot as plt

        H_intervention = H_avert + H_takeover + H_signal
        H_end = self.horizon

        ### calculate indices of rollouts
        dones = self.dones[:, 1]
        done_indices = [-1] + np.arange(len(dones))[dones].tolist()
        rollout_indices = [np.arange(done_indices[i]+1, done_indices[i+1] + 1) for i in range(len(done_indices) - 1)]
        rollout_indices = [indices for indices in rollout_indices if len(indices) >= H_intervention]

        ### keep track of
        steps_autonomous, steps_human = 0, 0
        num_crashes, num_crashes_averted = 0, 0

        ### for each rollout
        for idxs in rollout_indices:
            preds = self.preds[idxs]
            labels = self.labels[idxs]
            r_len = len(labels)

            is_human = False
            t = 0
            while t < r_len:
                ### pred_t is [num_bnn_samples, horizon]
                ### label_t is [horizon]
                pred_t = preds[t]
                label_t = labels[t]
                H_gt = (r_len - 1) - t if labels[-1, 0] else self.horizon

                H_pred = self.horizon - pred_t.mean(axis=0).sum()
                H_pred_std = np.sqrt(np.var(pred_t, axis=0).sum())

                # will_intervene = (H_pred < H_intervention)
                will_intervene = (H_pred - mean_thresh < H_intervention)
                if std_thresh is not None:
                    will_intervene = will_intervene or (H_pred_std > std_thresh)

                if H_gt < H_avert + H_takeover:
                    if is_human:
                        num_crashes_averted += 1
                    else:
                        num_crashes += 1
                    break

                if will_intervene:
                    if is_human:
                        steps_human += 1
                        t += 1
                    else:
                        steps_autonomous += min(H_takeover, r_len - 1 - t)
                        t += min(H_takeover, r_len - 1 - t)
                        is_human = True
                else:
                    if is_human:
                        steps_human += min(H_signal, r_len - 1 - t)
                        t += min(H_signal, r_len - 1 - t)
                        is_human = False
                    else:
                        steps_autonomous += 1
                        t += 1

            if t > r_len:
                print(t)
                print(r_len)
                raise Exception('This should never happen')


        frac_autonomous = steps_autonomous / float(steps_autonomous + steps_human)
        frac_crashes_averted = num_crashes_averted / float(num_crashes + num_crashes_averted)

        if save_dir is not None:
            f, ax = plt.subplots(1, 1)
            bar_auto = ax.bar([1], [frac_autonomous], width=0.4)
            bar_crash = ax.bar([2], [frac_crashes_averted], width=0.4)

            # ax.legend((bar_auto[0], bar_crash[0]), ('Time autonomous', 'Crashes averted'))

            ax.set_ylim([0, 1.05])
            plt.yticks(np.linspace(0, 1., 11))
            plt.xticks([1, 2], ('Time\nautonomous', 'Crashes\naverted'))
            ax.set_ylabel('Fraction')
            ax.set_title('plot online human switching ({0} rollouts)'.format(len(rollout_indices)))

            f.tight_layout()

            f.savefig(os.path.join(save_dir, 'plot_online_switching.png'), dpi=100, bbox_inches='tight')
            plt.close(f)

        return frac_autonomous, frac_crashes_averted

    def plot_online_switching_rollouts(self, H_avert=4, H_takeover=4, H_signal=4, save_dir=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        assert(save_dir is not None)

        H_intervention = H_avert + H_takeover + H_signal
        H_end = self.horizon

        ### calculate indices of rollouts
        dones = self.dones[:, 1]
        done_indices = [-1] + np.arange(len(dones))[dones].tolist()
        rollout_indices = [np.arange(done_indices[i]+1, done_indices[i+1] + 1) for i in range(len(done_indices) - 1)]
        rollout_indices = [indices for indices in rollout_indices if len(indices) >= H_intervention]

        ### keep track of
        colors_dict = {
            'auto': 'b',
            'switching_to_human': 'b',
            'human': 'gray',
            'switching_to_auto': 'gray',
            'crash': 'r',
            'crash_averted': 'g'
        }
        markers_dict = {
            'auto': 'o',
            'switching_to_human': 'o',
            'human': 'o',
            'switching_to_auto': 'o',
            'crash': '^',
            'crash_averted': '^'
        }
        rollouts_colors = []
        rollouts_markers = []
        rollouts_positions = []

        steps_autonomous, steps_human = 0, 0
        num_crashes, num_crashes_averted = 0, 0

        ### for each rollout
        for idxs in rollout_indices:
            preds = self.preds[idxs]
            labels = self.labels[idxs]
            env_infos = self.env_infos[idxs]
            r_len = len(labels)

            rollouts_positions.append(np.array([ei[0]['pos'][:2] for ei in env_infos]))
            assert (rollouts_positions[-1].shape == (r_len, 2))

            r_colors = []
            r_markers = []
            is_human = False
            t = 0
            while t < r_len:
                ### pred_t is [num_bnn_samples, horizon]
                ### label_t is [horizon]
                pred_t = preds[t]
                label_t = labels[t]
                H_gt = (r_len - 1) - t if labels[-1, 0] else self.horizon

                H_pred = self.horizon - pred_t.mean(axis=0).sum()
                will_intervene = (H_pred < H_intervention)

                if H_gt < H_avert + H_takeover:
                    if is_human:
                        num_crashes_averted += 1
                        r_colors += [colors_dict['crash_averted']] * (r_len - t)
                        r_markers += [markers_dict['crash_averted']] * (r_len - t)
                    else:
                        num_crashes += 1
                        r_colors += [colors_dict['crash']] * (r_len - t)
                        r_markers += [markers_dict['crash']] * (r_len - t)
                    break

                if will_intervene:
                    if is_human:
                        steps_human += 1
                        t += 1
                        r_colors += [colors_dict['human']]
                        r_markers += [markers_dict['human']]
                    else:
                        steps_autonomous += H_takeover
                        t += H_takeover
                        is_human = True
                        r_colors += [colors_dict['switching_to_human']] * H_takeover
                        r_markers += [markers_dict['human']] * H_takeover
                else:
                    if is_human:
                        steps_human += H_signal
                        t += H_signal
                        is_human = False
                        r_colors += [colors_dict['switching_to_auto']] * H_signal
                        r_markers += [markers_dict['human']] * H_signal
                    else:
                        steps_autonomous += 1
                        t += 1
                        r_colors += [colors_dict['auto']]
                        r_markers += [markers_dict['human']]

            if t >= r_len:
                raise Exception('This should never happen')

            assert (len(r_colors) == r_len)
            rollouts_colors.append(np.array(r_colors))
            rollouts_markers.append(np.array(r_markers))

        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i, (r_colors, r_markers, r_positions) in enumerate(zip(rollouts_colors, rollouts_markers, rollouts_positions)):
            x = r_positions[:, 0]
            y = r_positions[:, 1]
            for marker in set(markers_dict.values()):
                idxs = (r_markers == marker)
                ax.scatter(x[idxs], y[idxs], s=5, c=r_colors[idxs], marker=marker)
            ax.set_xlim((-23, 23))
            ax.set_ylim((-23, 23))

            labels = ['auto', 'human', 'crash', 'crash_averted']
            patches = [
                mpatches.Patch(color=colors_dict[label], label=label)
                for label in labels]
            ax.legend(patches, labels, loc='center')

            f.savefig(os.path.join(save_dir, 'plot_online_switching_rollouts_{0:03d}.png'.format(i)),
                      dpi=200, bbox_inches='tight')

            ax.cla()

        return f

    def plot_online_switching_mean(self, H_avert=4, H_takeover=4, H_signal=4, save_dir=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

        assert (save_dir is not None)
        pkl_fname = os.path.join(save_dir, 'plot_online_switching_mean_Ha{0}_Ht{1}_Hs{2}.pkl'.format(H_avert, H_takeover, H_signal))

        if not os.path.exists(pkl_fname):
            means = np.linspace(0, 5, 1000)
            means_used, frac_autonomous_list, frac_crashes_averted_list = [], [], []
            for mean in means:
                frac_autonomous, frac_crashes_averted = self.plot_online_switching(H_avert=H_avert,
                                                                                   H_takeover=H_takeover,
                                                                                   H_signal=H_signal,
                                                                                   mean_thresh=mean)

                if len(frac_autonomous_list) == 0 or abs(frac_autonomous - frac_autonomous_list[-1]) > 1e-3:
                    means_used.append(mean)
                    frac_autonomous_list.append(frac_autonomous)
                    frac_crashes_averted_list.append(frac_crashes_averted)

            mypickle.dump({'means_used': means_used,
                           'frac_autonomous_list': frac_autonomous_list,
                           'frac_crashes_averted_list': frac_crashes_averted_list},
                          pkl_fname)
        else:
            d = mypickle.load(pkl_fname)
            means_used = d['means_used']
            frac_autonomous_list = d['frac_autonomous_list']
            frac_crashes_averted_list = d['frac_crashes_averted_list']

        means_used.insert(0, means_used[0])
        frac_autonomous_list.insert(0, 1.0)
        frac_crashes_averted_list.insert(0, frac_crashes_averted_list[0])

        f, ax = plt.subplots(1, 1)

        ax.plot(frac_autonomous_list, 1 - np.array(frac_crashes_averted_list), color='b')

        ax.set_xlim((0, 1.05))
        ax.set_ylim((0, 1.05))
        plt.xticks(np.linspace(0., 1., 11))
        plt.yticks(np.linspace(0., 1., 11))

        # Customize the major grid
        ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')

        ax.set_xlabel('Fraction autonomous')
        ax.set_ylabel('Fraction crashes')
        ax.set_title('Autonomy vs crashes averted with mean thresholding')

        if save_dir:
            f.savefig(os.path.join(save_dir, 'plot_online_switching_mean.png'), dpi=200, bbox_inches='tight')

        return f

    def plot_online_switching_std(self, H_avert=4, H_takeover=4, H_signal=4, save_dir=None):
        import matplotlib.pyplot as plt

        assert (save_dir is not None)
        pkl_fname = os.path.join(save_dir, 'plot_online_switching_std_Ha{0}_Ht{1}_Hs{2}.pkl'.format(H_avert, H_takeover, H_signal))

        if not os.path.exists(pkl_fname):
            stds = np.linspace(0, 0.3, 1000)
            stds_used, frac_autonomous_list, frac_crashes_averted_list = [], [], []
            for std in stds:
                frac_autonomous, frac_crashes_averted = self.plot_online_switching(H_avert=H_avert,
                                                                                   H_takeover=H_takeover,
                                                                                   H_signal=H_signal,
                                                                                   std_thresh=std)

                if len(frac_autonomous_list) == 0 or abs(frac_autonomous - frac_autonomous_list[-1]) > 1e-3:
                    stds_used.append(std)
                    frac_autonomous_list.append(frac_autonomous)
                    frac_crashes_averted_list.append(frac_crashes_averted)

            mypickle.dump({'stds_used': stds_used,
                           'frac_autonomous_list': frac_autonomous_list,
                           'frac_crashes_averted_list': frac_crashes_averted_list},
                          pkl_fname)
        else:
            d = mypickle.load(pkl_fname)
            stds_used = d['stds_used']
            frac_autonomous_list = d['frac_autonomous_list']
            frac_crashes_averted_list = d['frac_crashes_averted_list']


        stds_used.append(stds_used[-1])
        frac_autonomous_list.append(1.0)
        frac_crashes_averted_list.append(frac_crashes_averted_list[-1])

        f, ax = plt.subplots(1, 1)

        ax.plot(frac_autonomous_list, 1 - np.array(frac_crashes_averted_list), color='r')

        ax.set_xlim((0, 1.05))
        ax.set_ylim((0, 1.05))
        plt.xticks(np.linspace(0., 1., 11))
        plt.yticks(np.linspace(0., 1., 11))

        # Customize the major grid
        ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')

        ax.set_xlabel('Fraction autonomous')
        ax.set_ylabel('Fraction crashes')
        ax.set_title('Autonomy vs crashes averted with std thresholding')

        if save_dir:
            f.savefig(os.path.join(save_dir, 'plot_online_switching_std.png'), dpi=200, bbox_inches='tight')

        return f


    def plot_online_decision(self, H_avert=2, H_headsup_min=1, H_headsup_window=2, save_dir=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        ### calculate indices of rollouts
        dones = self.dones[:, 1]
        done_indices = [-1] + np.arange(len(dones))[dones].tolist()
        rollout_indices = [np.arange(done_indices[i]+1, done_indices[i+1] + 1) for i in range(len(done_indices) - 1)]
        rollout_indices = [indices for indices in rollout_indices if len(indices) >= H_avert + H_headsup_min + H_headsup_window]

        ### pct thresh std
        # pct_thresh = 0.5
        # train_preds_std_avg = self.train_preds.std(axis=1).mean(axis=1).ravel()
        # std_thresh = np.percentile(train_preds_std_avg, pct_thresh)

        ### keep track of
        H_gts = []
        H_preds = []

        ### for each rollout
        for idxs in rollout_indices:
            preds = self.preds[idxs]
            labels = self.labels[idxs]
            r_len = len(labels)

            # print('')

            ### for each timestep
            for t, (pred_t, label_t) in enumerate(zip(preds, labels)):
                ### pred_t is [num_bnn_samples, horizon]
                ### label_t is [horizon]

                H_intervention_low = H_avert + H_headsup_min
                H_intervention_high = H_avert + H_headsup_min + H_headsup_window

                H_gt = (r_len - 1) - t if labels[-1, 0] else self.horizon

                H_pred = self.horizon - np.percentile(pred_t, 95., axis=0).sum()

                # H_pred = self.horizon - pred_t.mean(axis=0).sum()
                # # H_pred_std = pred_t.std(axis=0).sum()
                # H_pred_std = np.sqrt(np.var(pred_t, axis=0).sum())

                # print('t: {0:02d} | H_gt: {1:02d} | H_pred: {2:03.1f}'.format(t, H_gt, H_pred))

                assert((0 <= H_pred) and (H_pred <= self.horizon))

                # will_intervene = (H_pred < H_intervention_high)
                will_intervene = (H_pred < H_intervention_high)
                # will_intervene = will_intervene or (H_pred_std > 0.05)
                break_now = False

                if H_gt == 0:
                    H_pred = 0 # record that it messed up
                    break_now = True

                if will_intervene:
                    break_now = True

                if break_now:
                    H_gts.append(H_gt)
                    H_preds.append(H_pred)
                    break

            else:
                print('Did not crash')

        H_gts = np.array(H_gts)
        H_preds = np.array(H_preds)

        colors = ['r'] * H_avert + ['m'] * H_headsup_min + ['g'] * H_headsup_window + ['b'] * (self.horizon + 1 - H_avert - H_headsup_min - H_headsup_window)
        bins = np.zeros(self.horizon + 1, dtype=float)
        for H_gt, H_pred in zip(H_gts, H_preds):
            bins[int(np.floor(np.clip(H_gt, 0, self.horizon + 0.5)))] += 1
        bins /= bins.sum()

        f, ax = plt.subplots(1, 1)
        ax.bar(np.arange(len(bins)), bins, color=colors)
        plt.xticks(np.arange(0, len(bins) + 1))


        labels = ['Crash', 'Late', 'Perfect', 'Early']
        colors = ['r', 'm', 'g', 'b']
        thresholds = [H_avert, H_avert + H_headsup_min, H_avert + H_headsup_min + H_headsup_window, self.horizon]


        ax.set_ylim((0, 1.))
        ax.vlines(np.array(thresholds) - 0.5, 0, 1., color=colors, linestyle='--')


        patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in zip(labels, colors)]
        ax.legend(patches, labels, bbox_to_anchor=(1.2, 1.05))
        ax.set_xlabel('When will algorithm intervene (timesteps)')
        ax.set_ylabel('Fraction')
        ax.set_title('plot online decision making ({0} rollouts)'.format(len(rollout_indices)))

        f.tight_layout()

        if save_dir:
            f.savefig(os.path.join(save_dir, 'plot_online_decision.png'), dpi=100, bbox_inches='tight')
            plt.close(f)

        return f