import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class BnnPlotter(object):

    def __init__(self, preds, labels, train_preds=None, dones=None):
        """
        preds: num_replays x num_bnn_samples x action_len-1
        labels: num_replays x action_len
        :return:
        """
        self.preds = preds
        self.labels = labels
        self.dones = dones
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

    def plot_online_decision(self, H_avert=6, H_headsup_min=4, H_headsup_window=4):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        ### calculate indices of rollouts
        dones = self.dones[:, 1]
        done_indices = [-1] + np.arange(len(dones))[dones].tolist()
        rollout_indices = [np.arange(done_indices[i]+1, done_indices[i+1] + 1) for i in range(len(done_indices) - 1)]
        rollout_indices = [indices for indices in rollout_indices if len(indices) >= H_avert + H_headsup_min + H_headsup_window]

        ### pct thresh std
        pct_thresh = 0.5
        train_preds_std_avg = self.train_preds.std(axis=1).mean(axis=1).ravel()
        std_thresh = np.percentile(train_preds_std_avg, pct_thresh)

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

                H_gt = (r_len - 1) - t if labels[-1, 0] else self.horizon
                H_pred = self.horizon - pred_t.mean(axis=0).sum()
                H_pred_std = pred_t.std(axis=0).mean()

                # print('t: {0:02d} | H_gt: {1:02d} | H_pred: {2:03.1f}'.format(t, H_gt, H_pred))

                assert((0 <= H_pred) and (H_pred <= self.horizon))

                H_intervention_low = H_avert + H_headsup_min
                H_intervention_high = H_avert + H_headsup_min + H_headsup_window

                will_intervene = (H_pred < H_intervention_high) #or (H_pred_std > -1)
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

        ax.vlines(np.array(thresholds) - 0.5, 0, max(bins), color=colors, linestyle='--')


        patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in zip(labels, colors)]
        ax.legend(patches, labels, bbox_to_anchor=(1.2, 1.05))
        ax.set_xlabel('When will algorithm intervene (timesteps)')
        ax.set_ylabel('Fraction')
        ax.set_title('plot online decision making ({0} rollouts)'.format(len(rollout_indices)))

        f.tight_layout()

        return f