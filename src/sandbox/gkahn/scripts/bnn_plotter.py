import os
import numpy as np
import matplotlib.pyplot as plt

class BnnPlotter(object):

    def __init__(self, preds, labels):
        """
        preds: num_replays x num_bnn_samples x action_len-1
        labels: num_replays x action_len
        :return:
        """
        self.preds = preds
        self.labels = labels
        self.num_replays = preds.shape[0]
        self.num_bnn_samples = preds.shape[1]
        self.num_time_steps_pred = preds.shape[2]
        self.num_time_steps_replay = labels.shape[1]
        assert preds.shape[0] == labels.shape[0]

    def save_all_plots(self, dir, plot_types=None):
        if plot_types is None:
            plot_types = ['plot_hist_individual_predictions',
                          'plot_hist_time_to_collision',
                          'plot_scatter_time_to_collision',
                          'plot_pred_timeseries']

        for plot_type in plot_types:
            plot_method = self._get_plot_method(plot_type)
            fig, ax = plot_method()
            file_path = os.path.join(dir, plot_type + '.png')
            fig.savefig(file_path, bbox_inches='tight')
            plt.close(fig)


    def show(self, plot_type):
        plot_method = self._get_plot_method(plot_type)
        plot_method()
        plt.show(block=False)


    def _get_plot_method(self, plot_type):
        return getattr(self, plot_type)


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
        return fig, axs


    def _predict_collision(self):
        return np.mean(self.preds, axis=(1,2))


    def plot_predtruth(self):

        collision_truth = np.any(self.labels, axis=1).astype(float)  # num_replays
        collision_prediction = self._predict_collision(self.preds)  # num_replays

        # sort into ascending order of prediction
        pred, truth = zip(*sorted(zip(collision_prediction, collision_truth), key=lambda x: x[0]))
        cumulative_truth = np.cumsum(truth) / len(truth)

        fig, ax = plt.plot(pred, cumulative_truth)
        return fig, ax


    def plot_scatter_time_to_collision(self):
        truth = self.num_time_steps_replay - np.sum(self.labels, axis=1)  # num_replays
        pred_mu = self.num_time_steps_pred - np.sum(np.mean(self.preds, axis=1), axis=1)  # num_replays
        pred_sigma = np.sqrt(np.sum(np.var(self.preds, axis=1), axis=1))  # num_replays
        pred_mu_minus_truth = pred_mu - truth
        fig, ax = plt.subplots(1, 1)
        ax.scatter(pred_mu_minus_truth, pred_sigma, alpha=0.5)
        plt.title("Prediction minus truth of time steps until collision")
        plt.xlabel("Expectation")
        plt.ylabel("Standard deviation")
        return fig, ax


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
        return fig, ax


    def plot_hist_individual_predictions(self):
        truth = self.labels[:,:-1].reshape(-1)  # num_replays * num_time_steps_pred
        pred = np.mean(self.preds, axis=1).reshape(-1)  # num_replays * num_time_steps_pred
        pts = list(zip(pred, truth))
        pred_when_truth_collision = [pt[0] for pt in pts if pt[1] != 0.]
        pred_when_truth_no_collision = [pt[0] for pt in pts if pt[1] == 0.]

        fig, ax = plt.subplots(1, 1)
        ax.hist(pred_when_truth_collision, bins=50, alpha=0.5, label='Pred when truth=Collision')
        ax.hist(pred_when_truth_no_collision, bins=50,  alpha=0.5, label='Pred when truth=No Collision')
        plt.legend(loc='upper right')
        plt.xlabel("probability")
        return fig, ax


    def plot_roc(self):
        truth = self.labels[:,:-1].reshape(-1)  # num_replays * num_time_steps_pred
        pred = np.mean(self.preds, axis=1).reshape(-1)  # num_replays * num_time_steps_pred
        pts = list(zip(pred, truth))
        _, truth = list(zip(*sorted(pts, key=lambda x: x[0])))
        truth = abs(np.array(truth))
        num = len(truth)
        num_pos = sum(truth)
        num_neg = num - num_pos
        num_until_p = 1. + np.arange(num)
        num_pos_until_p = np.cumsum(truth)
        num_neg_until_p = num_until_p - num_pos_until_p
        num_pos_after_p = num_pos - num_pos_until_p
        num_neg_after_p = num_neg - num_neg_until_p

        false_pos_rate = num_neg_after_p / num_neg
        true_pos_rate = num_pos_after_p / num_pos

        fig, ax = plt.plot(false_pos_rate, true_pos_rate, color='darkorange')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        return fig, ax


    def plot_scatter_individual_predictions(self):
        truth = self.labels[:,:-1].reshape(-1)  # num_replays * num_time_steps_pred
        pred = np.mean(self.preds, axis=1).reshape(-1)  # num_replays * num_time_steps_pred
        sigma = np.std(self.preds, axis=1).reshape(-1)  # num_replays * num_time_steps_pred
        pts = list(zip(pred, sigma, truth))
        pred_when_truth_collision = [pt[0] for pt in pts if pt[2] != 0.]
        sigma_when_truth_collision = [pt[1] for pt in pts if pt[2] != 0.]
        pred_when_truth_no_collision = [pt[0] for pt in pts if pt[2] == 0.]
        sigma_when_truth_no_collision = [pt[1] for pt in pts if pt[2] == 0.]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pred_when_truth_collision, sigma_when_truth_collision, alpha=0.5, label='Pred when truth=Collision')
        ax.scatter(pred_when_truth_no_collision, sigma_when_truth_no_collision, alpha=0.5, label='Pred when truth=No Collision')
        plt.legend(loc='upper right')
        plt.xlabel("Expected Probability of Collision")
        plt.ylabel("Sigma Probability of Collision")
        return fig, ax
