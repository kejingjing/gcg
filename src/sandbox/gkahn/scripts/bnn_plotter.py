import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

        print(means_used)
        print(frac_autonomous_list)
        print(frac_crashes_averted_list)

        means_used = [0.0, 0.12012012012012012, 0.19519519519519518, 0.25525525525525528, 0.31031031031031031, 0.42542542542542544, 0.62562562562562563, 0.75575575575575571, 0.85585585585585588, 0.93093093093093093, 0.96596596596596596, 1.0110110110110111, 1.0560560560560561, 1.1011011011011012, 1.1611611611611612, 1.1911911911911912, 1.2862862862862863, 1.3213213213213213, 1.3863863863863863, 1.4614614614614614, 1.5265265265265264, 1.6016016016016015, 1.6666666666666667, 1.7517517517517518, 1.8518518518518519, 1.8968968968968969, 1.9419419419419419, 1.991991991991992, 2.0320320320320322, 2.0770770770770772, 2.1371371371371373, 2.1771771771771773, 2.2272272272272273, 2.2622622622622623, 2.3073073073073074, 2.3523523523523524, 2.4124124124124124, 2.4624624624624625, 2.5225225225225225, 2.5925925925925926, 2.6326326326326326, 2.6826826826826826, 2.7227227227227226, 2.7877877877877877, 2.8378378378378377, 2.8728728728728727, 2.9079079079079078, 2.9329329329329328, 2.9579579579579578, 2.9829829829829828, 3.0080080080080078, 3.0380380380380378, 3.0630630630630629, 3.0780780780780779, 3.1031031031031029, 3.1281281281281279, 3.1481481481481479, 3.1731731731731729, 3.198198198198198, 3.2282282282282284, 3.2432432432432434, 3.2732732732732734, 3.2982982982982985, 3.3183183183183185, 3.3433433433433435, 3.3633633633633635, 3.3833833833833835, 3.4234234234234235, 3.4584584584584586, 3.4834834834834836, 3.5085085085085086, 3.5235235235235236, 3.5485485485485486, 3.5785785785785786, 3.5985985985985987, 3.6186186186186187, 3.6336336336336337, 3.6586586586586587, 3.6736736736736737, 3.6936936936936937, 3.7087087087087087, 3.7237237237237237, 3.7437437437437437, 3.7487487487487487, 3.7787787787787788, 3.7937937937937938, 3.8138138138138138, 3.8238238238238238, 3.8338338338338338, 3.8388388388388388, 3.8488488488488488, 3.8588588588588588, 3.8688688688688688, 3.8788788788788788, 3.8888888888888888, 3.8938938938938938, 3.9039039039039038, 3.9089089089089089, 3.9139139139139139, 3.9189189189189189, 3.9239239239239239, 3.9289289289289289, 3.9339339339339339, 3.9389389389389389, 3.9439439439439439, 3.9489489489489489, 3.9539539539539539, 3.9589589589589589, 3.9639639639639639, 3.9689689689689689, 3.9739739739739739, 3.9789789789789789, 3.9839839839839839, 3.9889889889889889, 3.9939939939939939, 3.9989989989989989, 4.0040040040040044]
        frac_autonomous_list = [0.929844403317568, 0.9287501285170842, 0.9276191781760855, 0.926551735956404, 0.925489255235288, 0.9243582273708744, 0.9233300202214073, 0.922159383033419, 0.9210995338634494, 0.9200946112710818, 0.918997668997669, 0.9179777206512425, 0.9169151670951157, 0.9157811750188524, 0.9146500308493865, 0.9136128346645641, 0.9125158547872887, 0.9114188749100134, 0.910321895032738, 0.909296585767174, 0.9082650577628467, 0.9071027012203483, 0.9060057589469355, 0.9048745372274785, 0.9037808932917424, 0.9025401940283158, 0.901306091666381, 0.9000445617523052, 0.8989133788091729, 0.8977479175950365, 0.8967195694649162, 0.8956912213347958, 0.8945257601206595, 0.8934018851756641, 0.8923467114508002, 0.8912156835863866, 0.8899893759210391, 0.8888013158796518, 0.887739017202385, 0.8867336097878611, 0.8853284896672264, 0.8842660817711367, 0.8832036738750472, 0.8821150748774887, 0.8809842020492786, 0.8799520219328307, 0.8788553803975325, 0.8778272789581906, 0.876525017135024, 0.87538983515542, 0.8743317340644277, 0.8732008224811515, 0.8720356408498972, 0.8709732693625771, 0.8698766278272789, 0.8686771761480466, 0.8674434544208361, 0.8663810829335161, 0.865318711446196, 0.864256339958876, 0.8632531359243265, 0.8621221468229487, 0.8608540681335253, 0.859688806635136, 0.8585920899307697, 0.8573240112413463, 0.8562859884836852, 0.8551206471072114, 0.8536326250856752, 0.852628692850778, 0.8511790512750206, 0.8499108856594462, 0.8487455442829723, 0.847717301891966, 0.846586235261859, 0.8452495201535508, 0.8441816623821765, 0.8430505569837189, 0.8419194515852614, 0.8407431529153669, 0.8392006307201865, 0.8381722825900662, 0.8369837189374465, 0.8358868894601542, 0.8347787337606691, 0.8336361454869562, 0.8323334818826917, 0.8305166089609544, 0.8288711391450413, 0.8276488533918349, 0.8265176704487026, 0.8253016177680285, 0.823485056210584, 0.8219769673704415, 0.8204627249357327, 0.818783204798629, 0.8165552699228792, 0.815184233076264, 0.8132775816567844, 0.811358261644446, 0.81029578092333, 0.8089051895523411, 0.8061972989648317, 0.8043463357784328, 0.8022211558236786, 0.800712963597724, 0.7973195310893261, 0.794885857270172, 0.7924250214224507, 0.7882433590402742, 0.7825833647486206, 0.7763649449909175, 0.7670927721991844, 0.752999245903887, 0.7332396490265972, 0.6479341676667238, 0.007684391080617496]
        frac_crashes_averted_list = [0.36585365853658536, 0.34146341463414637, 0.34146341463414637, 0.36585365853658536, 0.36585365853658536, 0.36585365853658536, 0.36585365853658536, 0.36585365853658536, 0.3902439024390244, 0.3902439024390244, 0.3902439024390244, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.4634146341463415, 0.4634146341463415, 0.4634146341463415, 0.4634146341463415, 0.43902439024390244, 0.43902439024390244, 0.4634146341463415, 0.4634146341463415, 0.4634146341463415, 0.4634146341463415, 0.4634146341463415, 0.4634146341463415, 0.4634146341463415, 0.4878048780487805, 0.4878048780487805, 0.5121951219512195, 0.4878048780487805, 0.4878048780487805, 0.5121951219512195, 0.5121951219512195, 0.5121951219512195, 0.5121951219512195, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5121951219512195, 0.5121951219512195, 0.5121951219512195, 0.5365853658536586, 0.5365853658536586, 0.5365853658536586, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5853658536585366, 0.5853658536585366, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5853658536585366, 0.5853658536585366, 0.5853658536585366, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5609756097560976, 0.5853658536585366, 0.6341463414634146, 0.6585365853658537, 0.6585365853658537, 0.7560975609756098, 0.8292682926829268, 1.0]

        f, ax = plt.subplots(1, 1)

        ax.plot(means_used, frac_crashes_averted_list, color='g', label='Crashes averted')
        ax.plot(means_used, frac_autonomous_list, color='b', label='Time spent autonomous')

        ax.set_ylim((0, 1.05))
        plt.yticks(np.linspace(0., 1., 11))

        ax.legend(loc='lower right')
        ax.set_xlabel('Mean Threshold')
        ax.set_ylabel('Fraction')
        ax.set_title('Autonomy vs crashes averted with mean thresholding')

        if save_dir:
            f.savefig(os.path.join(save_dir, 'plot_online_switching_mean.png'), dpi=200, bbox_inches='tight')

        return f

    def plot_online_switching_std(self, H_avert=4, H_takeover=4, H_signal=4, save_dir=None):
        import matplotlib.pyplot as plt

        # stds = np.linspace(0, 0.3, 1000)
        # stds_used, frac_autonomous_list, frac_crashes_averted_list = [], [], []
        # for std in stds:
        #     frac_autonomous, frac_crashes_averted = self.plot_online_switching(H_avert=H_avert,
        #                                                                        H_takeover=H_takeover,
        #                                                                        H_signal=H_signal,
        #                                                                        std_thresh=std)
        #
        #     if len(frac_autonomous_list) == 0 or abs(frac_autonomous - frac_autonomous_list[-1]) > 1e-3:
        #         stds_used.append(std)
        #         frac_autonomous_list.append(frac_autonomous)
        #         frac_crashes_averted_list.append(frac_crashes_averted)
        #
        # print(stds_used)
        # print(frac_autonomous_list)
        # print(frac_crashes_averted_list)

        stds_used = [0.0, 0.00030030030030030029, 0.00060060060060060057, 0.00090090090090090081, 0.0012012012012012011, 0.0015015015015015015, 0.0018018018018018016, 0.0021021021021021022, 0.0024024024024024023, 0.0027027027027027024, 0.003003003003003003, 0.0033033033033033031, 0.0036036036036036032, 0.0039039039039039038, 0.0042042042042042043, 0.0045045045045045045, 0.0048048048048048046, 0.0051051051051051047, 0.0054054054054054048, 0.0057057057057057058, 0.006006006006006006, 0.0063063063063063061, 0.0066066066066066062, 0.0069069069069069063, 0.0078078078078078076, 0.0081081081081081086, 0.0084084084084084087, 0.0087087087087087088, 0.0093093093093093091, 0.0096096096096096092, 0.0099099099099099093, 0.010210210210210209, 0.01081081081081081, 0.01111111111111111, 0.011711711711711712, 0.012312312312312312, 0.012912912912912912, 0.013213213213213212, 0.013813813813813813, 0.014414414414414413, 0.015015015015015015, 0.015915915915915915, 0.016516516516516516, 0.017117117117117116, 0.017417417417417418, 0.018018018018018018, 0.018618618618618618, 0.019219219219219218, 0.02012012012012012, 0.020720720720720721, 0.021621621621621619, 0.02222222222222222, 0.022822822822822823, 0.023423423423423424, 0.024024024024024024, 0.024624624624624624, 0.025225225225225224, 0.026126126126126126, 0.026726726726726727, 0.027627627627627625, 0.028828828828828826, 0.029729729729729728, 0.03063063063063063, 0.031531531531531529, 0.032432432432432434, 0.033333333333333333, 0.034834834834834835, 0.035435435435435432, 0.036336336336336338, 0.037237237237237236, 0.038438438438438437, 0.039939939939939939, 0.040840840840840838, 0.042642642642642642, 0.043543543543543541, 0.045045045045045043, 0.046246246246246243, 0.047147147147147142, 0.048348348348348349, 0.048948948948948946, 0.050450450450450449, 0.051651651651651649, 0.052252252252252253, 0.053453453453453453, 0.054954954954954956, 0.056456456456456451, 0.058858858858858859, 0.06006006006006006, 0.061861861861861857, 0.063063063063063057, 0.065465465465465458, 0.067267267267267269, 0.068468468468468463, 0.071171171171171166, 0.072372372372372373, 0.073873873873873869, 0.075075075075075076, 0.076576576576576572, 0.077777777777777779, 0.079579579579579576, 0.082282282282282279, 0.08468468468468468, 0.087987987987987987, 0.089789789789789784, 0.091591591591591581, 0.093993993993993996, 0.096996996996996987, 0.10240240240240239, 0.1057057057057057, 0.11201201201201201, 0.11801801801801801, 0.12642642642642643, 0.13393393393393394, 0.14174174174174173, 0.17087087087087086, 0.19699699699699699]
        frac_autonomous_list = [0.007684391080617496, 0.6936208137661536, 0.7248234729553712, 0.7439851943244911, 0.7549957155098543, 0.7617905127502056, 0.767290424292275, 0.7719171978888203, 0.7759270683391596, 0.7786955478630428, 0.7819172635980396, 0.7848869088416723, 0.7878881348961546, 0.7898560657984921, 0.792110764590973, 0.7935773528000548, 0.7949482486805127, 0.7969017753101652, 0.7985056722761079, 0.800527813003393, 0.8022757651574871, 0.803338245878603, 0.8049758404441246, 0.806640625, 0.8090881052739797, 0.8101912137619081, 0.8113220478377082, 0.8129668974025084, 0.8145432115687753, 0.8156271418779987, 0.8166398245553919, 0.8177176148046608, 0.8196243745287546, 0.8210980876002467, 0.8225436101305733, 0.8247712395901162, 0.8260228908231101, 0.827599204989377, 0.8286615036666438, 0.8303865131578947, 0.8317913925438597, 0.8341500188465888, 0.8354521467977932, 0.8372340060994414, 0.8383648014254874, 0.8399300914978924, 0.8409581577053562, 0.8421629771777123, 0.8436707559454458, 0.8453156055102461, 0.8473530923419564, 0.8488949802980983, 0.850328947368421, 0.8513467205811802, 0.8527022858905378, 0.8541074060111724, 0.8556496110216252, 0.8570694263587143, 0.8581317250359811, 0.8593701812699174, 0.8609464414213754, 0.8622438489479817, 0.8633404153245151, 0.8649371123067959, 0.866826593557231, 0.8682316655243317, 0.870125419779316, 0.8716331985470496, 0.8733721727210418, 0.8748457847840987, 0.875976696367375, 0.877129245638688, 0.878714143733507, 0.8801878062990507, 0.8815243839747764, 0.8826210630933206, 0.8838548271016827, 0.8852256759998629, 0.8862880838959526, 0.8873162205695877, 0.8883443572432229, 0.8894753075842216, 0.8905377154803111, 0.8919243161719339, 0.8931962296486718, 0.8944301628106255, 0.8955269922879178, 0.8967266495287061, 0.8979263067694945, 0.8989888603256212, 0.9000856898029135, 0.9013606607944614, 0.9025602358021729, 0.9042020839045791, 0.9054799684704753, 0.9065081051441105, 0.9076047842626547, 0.9088042770485624, 0.909866684944652, 0.9108887137128561, 0.911916920862323, 0.913050928781959, 0.9144590287535557, 0.9156585215394634, 0.9166838028651724, 0.9179176091575845, 0.9190143258619508, 0.9200424977722942, 0.921173486873672, 0.9222016587840154, 0.9232983754883817, 0.924561283246504, 0.9257926306769494, 0.9268209083119109, 0.9278908766879156, 0.9290218657892934]
        frac_crashes_averted_list = [1.0, 0.7073170731707317, 0.6829268292682927, 0.6829268292682927, 0.6585365853658537, 0.6585365853658537, 0.6585365853658537, 0.6585365853658537, 0.6585365853658537, 0.6097560975609756, 0.6097560975609756, 0.6097560975609756, 0.6097560975609756, 0.6097560975609756, 0.6097560975609756, 0.6097560975609756, 0.6097560975609756, 0.6097560975609756, 0.5853658536585366, 0.5853658536585366, 0.5853658536585366, 0.5853658536585366, 0.5609756097560976, 0.5365853658536586, 0.5121951219512195, 0.4878048780487805, 0.4878048780487805, 0.4878048780487805, 0.4878048780487805, 0.4634146341463415, 0.4878048780487805, 0.5121951219512195, 0.4878048780487805, 0.4878048780487805, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.43902439024390244, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.43902439024390244, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.3902439024390244, 0.3902439024390244, 0.3902439024390244, 0.36585365853658536, 0.36585365853658536, 0.34146341463414637, 0.34146341463414637, 0.3170731707317073, 0.3170731707317073, 0.2926829268292683, 0.2926829268292683, 0.2926829268292683, 0.3170731707317073, 0.3170731707317073, 0.3170731707317073, 0.3170731707317073, 0.3170731707317073, 0.3170731707317073, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.3170731707317073, 0.3170731707317073, 0.3170731707317073, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.34146341463414637, 0.36585365853658536, 0.36585365853658536, 0.3902439024390244, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.3902439024390244, 0.3902439024390244, 0.4146341463414634, 0.4146341463414634, 0.4146341463414634, 0.3902439024390244, 0.3902439024390244, 0.3902439024390244, 0.3902439024390244, 0.3902439024390244, 0.3902439024390244, 0.3902439024390244, 0.36585365853658536, 0.36585365853658536, 0.36585365853658536, 0.36585365853658536, 0.36585365853658536]

        f, ax = plt.subplots(1, 1)

        ax.semilogx(stds_used, frac_crashes_averted_list, color='g', label='Crashes averted')
        ax.semilogx(stds_used, frac_autonomous_list, color='b', label='Time spent autonomous')

        ax.set_ylim((0, 1.05))
        ax.set_xlim((0, max(stds_used)))
        plt.yticks(np.linspace(0., 1., 11))

        ax.legend(loc='lower right')
        ax.set_xlabel('Standard Deviation Threshold')
        ax.set_ylabel('Fraction')
        ax.set_title('Autonomy vs crashes averted with uncertainty thresholding')

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