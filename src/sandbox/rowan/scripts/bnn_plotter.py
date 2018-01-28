import matplotlib.pyplot as plt
import numpy as np
import os


class BnnPlotter(object):

    def __init__(self, preds, labels, env=None, env_infos=None, comparison_plotter=None):
        """
        preds: num_replays x num_bnn_samples x action_len-1
        labels: num_replays x action_len
        :return:
        """
        self.preds = preds
        self.labels = labels
        self.env = env
        self.env_infos = env_infos
        self.comparison_plotter = comparison_plotter
        self.num_replays = preds.shape[0]
        self.num_bnn_samples = preds.shape[1]
        self.num_time_steps_pred = preds.shape[2]
        self.num_time_steps_replay = labels.shape[1]
        assert preds.shape[0] == labels.shape[0]

        self.sigma_percentiles = None
        self.map_uncertainties = None

    def save_all_evaluation_plots(self, dir, env_name="UnknownEnv"):
        plot_types = ['plot_hist_individual_predictions',
                      'plot_hist_time_to_collision',
                      'plot_scatter_time_to_collision',
                      'plot_pred_timeseries']

        for plot_type in plot_types:
            fig = self._plot(plot_type)
            fig_name = env_name + plot_type
            self._save_fig(fig, dir, fig_name)


    def save_map_uncertainty_plot(self, dir, env_name="UnkownEnv", **kwargs):
        fig, _ = self.plot_map_uncertainty(**kwargs)
        fig_name = env_name + 'plot_map_uncertainty'
        self._save_fig(fig, dir, fig_name)


    def show(self, plot_type, **kwargs):
        self._plot(plot_type, **kwargs)
        plt.show(block=False)


    def _plot(self, plot_type, **kwargs):
        plot_method = getattr(self, plot_type)
        fig, ax = plot_method(**kwargs)
        return fig


    def _save_fig(self, fig, dir, fig_name="unnamed", close_fig=True):
        file_path = os.path.join(dir, fig_name + '.png')
        fig.savefig(file_path, bbox_inches='tight')
        if close_fig:
            plt.close(fig)


    def plot_pred_timeseries(self, num_replays_to_plot=10):
        num_replays_to_plot = min(num_replays_to_plot, self.num_replays)
        fig, axs = plt.subplots(num_replays_to_plot, self.num_time_steps_pred,
                                sharex=True, sharey=True, tight_layout=False, figsize=(20, 16))
        for i_replay in range(num_replays_to_plot):
            for i_time in range(self.num_time_steps_pred):
                ax = axs[i_replay, i_time]
                x = self.preds[i_replay, :, i_time]
                color = 'b' if self.labels[i_replay, i_time] == 0 else 'r'
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
        truth = self.num_time_steps_replay - np.sum(self.labels, axis=1)  # num_replays
        pred = self.num_time_steps_pred - np.sum(np.mean(self.preds, axis=1), axis=1)  # num_replays
        pred_minus_truth = pred - truth
        fig, ax = plt.subplots(1, 1)
        ax.hist(pred_minus_truth, 20)
        plt.xlabel("time steps (relative)")
        plt.title("predicted minus truth of time step collided")
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

    def _compute_map_uncertainties(self):
        if self.map_uncertainties is not None:
            return
        max_num_replays = 100000
        num_replays = len(self.env_infos)  # env_infos: num_replays x action_len-1
        num_replays = min(num_replays, max_num_replays)
        xs, ys, us = [], [] ,[]
        for i in range(num_replays):
            info = self.env_infos[i, 0]
            if not info:  # is empty after a collision
                continue
            pred_sigma = np.sqrt(np.sum(np.var(self.preds[i], axis=0)))  # num_replays
            x, y = info['pos'][:2]
            # coll = info['coll']
            xs.append(x)
            ys.append(y)
            us.append(pred_sigma)  # uncertainties

        xs = np.array(xs)
        ys = np.array(ys)
        us = np.array(us)
        self.map_uncertainties = {'xs': xs, 'ys': ys, 'us': us}

        low = np.percentile(us, 10)
        high = np.percentile(us, 90)
        self.sigma_percentiles = {'min': np.min(us), 'low': low, 'high': high, 'max': np.max(us)}

    def plot_map_uncertainty(self):
        """
        env_infos: num_replays x action_len-1
        :return:
        """
        self._compute_map_uncertainties()
        xs, ys, us = [self.map_uncertainties[i] for i in ['xs', 'ys', 'us']]

        # normalise roughly between 0 and 1
        if self.comparison_plotter is None:
            sigma_percentiles = self.sigma_percentiles
        else:
            sigma_percentiles = self.comparison_plotter.sigma_percentiles

        cm = plt.cm.get_cmap('coolwarm')
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        sc = ax.scatter(xs, ys, c=us, vmin=0, vmax=sigma_percentiles['high'], alpha=0.5, cmap=cm)
        if hasattr(self.env, 'cone_positions'):
            cone_positions = np.array(self.env.cone_positions)  # num_cones x 3 (xyz)
            ax.scatter(cone_positions[:,0], cone_positions[:,1], color='black', s=100)
        title = "Predictive time-to-collision-uncertainty (blue = more certain, red = more uncertain). " \
                "Sigma range [{0:.6f}, {1:.6f}]"
        plt.title(title.format(self.sigma_percentiles['low'], self.sigma_percentiles['high']))
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.colorbar(sc)
        return fig, ax

    def save_comparitive_decisions(self, dir):
        self._compute_map_uncertainties()
        self.comparison_plotter._compute_map_uncertainties()

        sigmas = []
        for plotter in [self, self.comparison_plotter]:
            for key in ['low', 'high', 'max']:
                sigmas.append(plotter.sigma_percentiles[key])
        sigmas.sort()
        sigma_decision_thresholds = []
        for i in range(len(sigmas)-1):
            sigma_decision_thresholds.append(np.linspace(sigmas[i], sigmas[i+1], 5))
        sigma_decision_thresholds = np.array(sigma_decision_thresholds).flatten()
        assert len(sigma_decision_thresholds) <= 1000  # don't save too many files by mistake

        cm = plt.cm.get_cmap('coolwarm')
        for i_sigma, sigma in enumerate(sigma_decision_thresholds):
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 16))
            axs[0].set_title("TrainingData. Blue = fine, Red = need help. Sigma threshold {0:.5f}".format(sigma))
            axs[1].set_title("HoldoutData. Blue = fine, Red = need help. Sigma threshold {0:.5f}".format(sigma))
            for i, plotter in enumerate([self.comparison_plotter, self]):
                us_thresholded = np.where(plotter.map_uncertainties['us'] < sigma, 0., 1.)
                axs[i].scatter(plotter.map_uncertainties['xs'], plotter.map_uncertainties['ys'],
                                    c=us_thresholded, vmin=0., vmax=1., alpha=0.1, cmap=cm)
                if hasattr(plotter.env, 'cone_positions'):
                    cone_positions = np.array(plotter.env.cone_positions)  # num_cones x 3 (xyz)
                    axs[i].scatter(cone_positions[:,0], cone_positions[:,1], color='black', s=100)
                axs[i].set_xlabel("X position")
                axs[i].set_ylabel("Y position")
            fig_name = 'comparitive_decisions{}'.format(i_sigma)
            self._save_fig(fig, dir, fig_name)
