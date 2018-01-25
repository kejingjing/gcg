import numpy as np

class BnnPlotter(object):

    @staticmethod
    def plot_dropout(preds, labels, num_sample=10):
        """
        outputs: num_dropout x sample_size x action_len-1
        rewards: sample_size x action_len
        :return:
        """
        import matplotlib.pyplot as plt
        num_sample = min(num_sample, preds.shape[1])
        num_time = preds.shape[2]
        i = 0
        for i_sample in range(num_sample):
            for i_time in range(num_time):
                x = preds[:, i_sample, i_time]
                i += 1
                plt.subplot(num_sample,num_time,i)
                color = 'b' if labels[i_sample, i_time] == 0 else 'r'
                plt.hist(x, 20, range=(0.,1.), color=color)
                if i_sample == 0:
                    plt.title("t = {}".format(i_time))
                if i_time == 0:
                    plt.ylabel("sample {}".format(i_sample+1))
        plt.show(block=False)

    @staticmethod
    def predict_collision(outputs):
        """
        outputs: num_dropout x sample_size x action_len-1
        :return:
        """
        return -np.mean(outputs, axis=(0,2))  # TODO: check this is a (negative) noramized probability

    @staticmethod
    def plot_predtruth(outputs, rewards):
        """
        outputs: num_dropout x sample_size x action_len-1
        rewards: sample_size x action_len
        :return:
        """
        import matplotlib.pyplot as plt
        collision_truth = np.any(rewards, axis=1).astype(float)  # sample_size
        collision_prediction = BnnPlotter.predict_collision(outputs)  # sample_size

        # sort into ascending order of prediction
        pred, truth = zip(*sorted(zip(collision_prediction, collision_truth), key=lambda x: x[0]))
        cumulative_truth = np.cumsum(truth) / len(truth)

        plt.plot(pred, cumulative_truth)
        plt.show(block=False)

    @staticmethod
    def plot_hist(outputs, rewards):
        """
        outputs: num_dropout x sample_size x action_len-1
        rewards: sample_size x action_len
        :return:
        """
        import matplotlib.pyplot as plt
        # time step to collision
        truth = rewards.shape[1] + np.sum(rewards, axis=1)  # sample_size
        pred = outputs.shape[2] + np.sum(np.mean(outputs, axis=0), axis=1)  # sample_size
        pred_minus_truth = pred - truth
        plt.hist(pred_minus_truth, 20)
        plt.xlabel("time steps (relative)")
        plt.title("predicted minus truth of time step collided")
        plt.show(block=False)

    @staticmethod
    def plot_hist_no_time_structure(outputs, rewards):
        """
        outputs: num_dropout x sample_size x action_len-1
        rewards: sample_size x action_len
        :return:
        """
        import matplotlib.pyplot as plt
        truth = -rewards[:,:-1].reshape(-1)  # sample_size * action_len-1
        pred = -np.mean(outputs, axis=0).reshape(-1)  # sample_size * action_len-1
        pts = list(zip(pred, truth))
        pred_when_truth_collision = [pt[0] for pt in pts if pt[1] != 0.] #list(filter(lambda x: x[1] != 0., pt))
        pred_when_truth_no_collision = [pt[0] for pt in pts if pt[1] == 0.]# list(filter(lambda x: x[1] == 0., pt))

        plt.hist(pred_when_truth_collision, bins=50, alpha=0.5, label='Pred when truth=Collision')
        plt.hist(pred_when_truth_no_collision, bins=50,  alpha=0.5, label='Pred when truth=No Collision')
        plt.legend(loc='upper right')
        plt.xlabel("probability")
        plt.show(block=False)

    @staticmethod
    def plot_roc(outputs, rewards):
        import matplotlib.pyplot as plt

        truth = -rewards[:,:-1].reshape(-1)  # sample_size * action_len-1
        pred = -np.mean(outputs, axis=0).reshape(-1)  # sample_size * action_len-1
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

        plt.plot(false_pos_rate, true_pos_rate, color='darkorange')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show(block=False)

    @staticmethod
    def plot_scatter_prob_and_sigma(outputs, rewards):
        """
        outputs: num_dropout x sample_size x action_len-1
        rewards: sample_size x action_len
        :return:
        """
        import matplotlib.pyplot as plt

        truth = -rewards[:,:-1].reshape(-1)  # sample_size * action_len-1
        pred = -np.mean(outputs, axis=0).reshape(-1)  # sample_size * action_len-1
        sigma = np.std(outputs, axis=0).reshape(-1)  # sample_size * action_len-1
        pts = list(zip(pred, sigma, truth))
        pred_when_truth_collision = [pt[0] for pt in pts if pt[2] != 0.]
        sigma_when_truth_collision = [pt[1] for pt in pts if pt[2] != 0.]
        pred_when_truth_no_collision = [pt[0] for pt in pts if pt[2] == 0.]
        sigma_when_truth_no_collision = [pt[1] for pt in pts if pt[2] == 0.]

        plt.scatter(pred_when_truth_collision, sigma_when_truth_collision, alpha=0.5, label='Pred when truth=Collision')
        plt.scatter(pred_when_truth_no_collision, sigma_when_truth_no_collision, alpha=0.5, label='Pred when truth=No Collision')
        plt.legend(loc='upper right')
        plt.xlabel("Expected Probability of Collision")
        plt.ylabel("Sigma Probability of Collision")
        plt.show(block=False)
