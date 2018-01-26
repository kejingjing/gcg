import os
import numpy as np
import matplotlib.pyplot as plt

class BnnPlotter(object):

    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels

    def save_all_plots(self, dir):

        plot_types = ['plot_hist_individual_predictions',
                      'plot_hist_time_to_collision',
                      'plot_scatter_time_to_collision',
                      'plot_pred_timeseries']

        for plot_type in plot_types:
            # print(plot_type)
            plot_method = self._get_plot_method(plot_type)
            fig, ax = plot_method()
            # ax.plot()
            file_path = os.path.join(dir, plot_type + '.png')
            fig.savefig(file_path, bbox_inches='tight')
            plt.close(fig)

    def show(self, plot_type):
        plot_method = self._get_plot_method(plot_type)
        plot_method()
        plt.show(block=False)

    def _get_plot_method(self, plot_type):
        return getattr(self, plot_type)

    def plot_pred_timeseries(self, num_sample=10):
        """
        preds: num_dropout x sample_size x action_len-1
        labels: sample_size x action_len
        :return:
        """
        num_sample = min(num_sample, self.preds.shape[1])
        num_time = self.preds.shape[2]
        fig, axs = plt.subplots(num_sample, num_time, sharex=True, sharey=True, tight_layout=False, figsize=(18,10))
        # fig.set_size_inches(18, 10)
        for i_sample in range(num_sample):
            for i_time in range(num_time):
                ax = axs[i_sample, i_time]
                x = self.preds[:, i_sample, i_time]
                color = 'b' if self.labels[i_sample, i_time] == 0 else 'r'
                ax.hist(x, 20, range=(0.,1.), color=color)
                if i_sample == num_sample - 1:
                    ax.set_xlabel("t = {}".format(i_time))
                if i_time == 0:
                    ax.set_ylabel("sample {}".format(i_sample+1))
        return fig, axs

    def predict_collision(self):
        """
        preds: num_dropout x sample_size x action_len-1
        :return:
        """
        return np.mean(self.preds, axis=(0,2))

    def plot_predtruth(self):
        """
        preds: num_dropout x sample_size x action_len-1
        labels: sample_size x action_len
        :return:
        """
        collision_truth = np.any(self.labels, axis=1).astype(float)  # sample_size
        collision_prediction = self.predict_collision(self.preds)  # sample_size

        # sort into ascending order of prediction
        pred, truth = zip(*sorted(zip(collision_prediction, collision_truth), key=lambda x: x[0]))
        cumulative_truth = np.cumsum(truth) / len(truth)

        fig, ax = plt.plot(pred, cumulative_truth)
        return fig, ax


    def plot_scatter_time_to_collision(self):
        """
        preds: num_dropout x sample_size x action_len-1
        labels: sample_size x action_len
        :return:
        """
        truth = self.labels.shape[1] - np.sum(self.labels, axis=1)  # sample_size
        pred_mu = self.preds.shape[2] - np.sum(np.mean(self.preds, axis=0), axis=1)  # sample_size
        pred_sigma = np.sqrt(np.sum(np.var(self.preds, axis=0), axis=1))  # sample_size
        pred_mu_minus_truth = pred_mu - truth
        fig, ax = plt.subplots(1, 1)
        ax.scatter(pred_mu_minus_truth, pred_sigma, alpha=0.5)
        plt.title("Prediction minus truth of time steps until collision")
        plt.xlabel("Expectation")
        plt.ylabel("Standard deviation")
        return fig, ax


    def plot_hist_time_to_collision(self):
        """
        preds: num_dropout x sample_size x action_len-1
        labels: sample_size x action_len
        :return:
        """
        truth = self.labels.shape[1] - np.sum(self.labels, axis=1)  # sample_size
        pred = self.preds.shape[2] - np.sum(np.mean(self.preds, axis=0), axis=1)  # sample_size
        pred_minus_truth = pred - truth
        fig, ax = plt.subplots(1, 1)
        ax.hist(pred_minus_truth, 20)
        plt.xlabel("time steps (relative)")
        plt.title("predicted minus truth of time step collided")
        return fig, ax


    def plot_hist_individual_predictions(self):
        """
        preds: num_dropout x sample_size x action_len-1
        labels: sample_size x action_len
        :return:
        """
        truth = self.labels[:,:-1].reshape(-1)  # sample_size * action_len-1
        pred = np.mean(self.preds, axis=0).reshape(-1)  # sample_size * action_len-1
        pts = list(zip(pred, truth))
        pred_when_truth_collision = [pt[0] for pt in pts if pt[1] != 0.] #list(filter(lambda x: x[1] != 0., pt))
        pred_when_truth_no_collision = [pt[0] for pt in pts if pt[1] == 0.]# list(filter(lambda x: x[1] == 0., pt))

        fig, ax = plt.subplots(1, 1)
        ax.hist(pred_when_truth_collision, bins=50, alpha=0.5, label='Pred when truth=Collision')
        ax.hist(pred_when_truth_no_collision, bins=50,  alpha=0.5, label='Pred when truth=No Collision')
        plt.legend(loc='upper right')
        plt.xlabel("probability")
        return fig, ax


    def plot_roc(self):
        truth = self.labels[:,:-1].reshape(-1)  # sample_size * action_len-1
        pred = np.mean(self.preds, axis=0).reshape(-1)  # sample_size * action_len-1
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

        # for p in pred:
        false_pos_rate = num_neg_after_p / num_neg
        true_pos_rate = num_pos_after_p / num_pos

        fig, ax = plt.plot(false_pos_rate, true_pos_rate, color='darkorange')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        return fig, ax


    def plot_scatter_individual_predictions(self):
        """
        preds: num_dropout x sample_size x action_len-1
        labels: sample_size x action_len
        :return:
        """
        truth = self.labels[:,:-1].reshape(-1)  # sample_size * action_len-1
        pred = np.mean(self.preds, axis=0).reshape(-1)  # sample_size * action_len-1
        sigma = np.std(self.preds, axis=0).reshape(-1)  # sample_size * action_len-1
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
