import os, itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

from gcg.policies.gcg_policy import GCGPolicy
from gcg.sampler.replay_pool import ReplayPool
from gcg.analyze.experiment import ExperimentGroup, MultiExperimentComparison
from gcg.data.logger import logger
from gcg.data import mypickle

CURR_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURR_DIR[:CURR_DIR.find('src/sandbox')], 'data')

def plot_test():
    label_params = [['exp', ('exp_name',)]]

    experiment_groups = [
        ExperimentGroup(os.path.join(DATA_DIR, 'sim_rccar/benchmarks/coll_dql'),
                        label_params=label_params,
                        plot={
                            'color': 'k'
                        }),
        ExperimentGroup(os.path.join(DATA_DIR, 'sim_rccar/benchmarks/coll_ours_bootstrap'),
                        label_params=label_params,
                        plot={
                            'color': 'b'
                        }),
        ExperimentGroup(os.path.join(DATA_DIR, 'sim_rccar/benchmarks/coll_ours'),
                        label_params=label_params,
                        plot={
                            'color': 'g'
                        }),
    ]

    mec = MultiExperimentComparison(experiment_groups)

    mec.plot_csv(['EvalEpisodeLengthMean'],
                 save_path=None,
                 plot_std=True,
                 avg_window=50,
                 xlim=None,
                 ylim=None)

def plot_rw_rccar_var001_var016_OLD(ckpt_itr=None):
    label_params =[['exp', ('exp_name',)]]

    experiment_groups = [
        ExperimentGroup(os.path.join(DATA_DIR, 'rw_rccar/var{0:03d}'.format(i)),
                        label_params=label_params,
                        plot={

                        }) for i in [19]
    ]

    mec = MultiExperimentComparison(experiment_groups)
    exps = mec.list

    ### plot length of the rollouts
    # f, axes = plt.subplots(1, len(exps), figsize=(32, 6), sharex=True, sharey=True)
    # for i, (exp, ax) in enumerate(zip(exps, axes)):
    #     rollouts = list(itertools.chain(*exp.train_rollouts))
    #     lengths = [len(r['dones']) for r in rollouts][:16]
    #     assert (len(lengths) == 16)
    #     steps = np.arange(len(lengths))
    #
    #     label = '{0}, height: {1}, color: {2}, H: {3}'.format(
    #         exp.name,
    #         exp.params['alg']['env'].split("'obs_shape': (")[1].split(',')[0],
    #         exp.params['alg']['env'].split(',')[-1].split(')})')[0],
    #         exp.params['policy']['H']
    #     )
    #
    #     ax.scatter(steps, lengths, color=cm.magma(i / len(exps)))
    #     ax.set_title(label)
    #     ax.legend()
    #
    # plt.tight_layout()
    # f.savefig('plots/rw-rccar/var001_016.png', bbox_inches='tight', dpi=100)

    ### plot policy on the rollouts

    # candidate actions
    for exp in exps:
        rollouts = mypickle.load(os.path.join(exp.folder, 'rosbag_rollouts_00.pkl'))['rollouts']
        rollouts = rollouts[::len(rollouts) // 16]
        # rollouts = list(itertools.chain(*exp.train_rollouts))[:16]

        tf_sess, tf_graph = GCGPolicy.create_session_and_graph(gpu_device=1, gpu_frac=0.6)

        with tf_sess.as_default(), tf_graph.as_default():
            exp.create_env()
            exp.create_policy()
            exp_ckpt_itr = exp.restore_policy(itr=ckpt_itr)

            K = 2048
            actions = np.random.uniform(*exp.env.action_space.bounds,
                                        size=(K, exp.policy.H + 1, exp.env.action_space.flat_dim))

            replay_pool = ReplayPool(
                env_spec=exp.env.spec,
                env_horizon=exp.env.horizon,
                N=exp.policy.N,
                gamma=1,
                size=int(1.1 * sum([len(r['dones']) for r in rollouts])),
                obs_history_len=exp.policy.obs_history_len,
                sampling_method='uniform'
            )

            step = 0
            outputs = []
            for i, r in enumerate(rollouts):
                r_len = len(r['dones'])
                outputs_i = []
                for j in range(r_len):
                    # evaluate and get output
                    observation = (r['observations'][j][0], np.empty([exp.policy.obs_history_len, 0]))
                    replay_pool.store_observation(step, observation)

                    encoded_observation = replay_pool.encode_recent_observation()

                    observation_im, observation_vec = encoded_observation
                    observations = (np.tile(observation_im, (K, 1, 1)), np.tile(observation_vec, (K, 1, 1)))

                    probcolls = exp.policy.get_model_outputs(observations, actions)
                    outputs_i.append(probcolls)

                    step += 1
                    replay_pool.store_effect(
                        r['actions'][j],
                        r['rewards'][j],
                        r['dones'][j],
                        None,
                        r['est_values'][j],
                        r['logprobs'][j]
                    )

                outputs.append(outputs_i)

        f, axes = plt.subplots(1, 2, figsize=(12, 8))
        imshow = None

        plot_folder = os.path.join(exp.folder, 'plot', 'ckpt_{0:03d}'.format(exp_ckpt_itr))
        os.makedirs(plot_folder, exist_ok=True)

        for i, (r_i, output_i) in enumerate(zip(rollouts, outputs)):
            for j, (obs, cost) in enumerate(zip(r_i['observations'], output_i)):
                obs_im, obs_vec = obs
                probcoll = -cost

                # plot image
                im = np.reshape(obs_im, exp.env.observation_im_space.shape)
                is_gray = (im.shape[-1] == 1)
                if is_gray:
                    im = im[:, :, 0]
                    color = 'Greys_r'
                else:
                    color=None

                if imshow is None:
                    imshow = axes[0].imshow(im, cmap=color)
                else:
                    imshow.set_data(im)

                # plot probcolls
                steers = actions[:, :-1, 0]
                angle_const = 0.5 * np.pi / 2.
                angles = angle_const * steers
                ys = np.cumsum(np.cos(angles), axis=1)
                xs = np.cumsum(-np.sin(angles), axis=1)
                sort_idxs = np.argsort(probcoll)

                xlim = (min(xs.min(), 0), max(xs.max(), 0))
                ylim = (min(ys.min(), -0.5), max(ys.max(), 0.5))
                min_probcoll = probcoll.min()
                max_probcoll = probcoll.max()

                keep = 10
                ys = ys[sort_idxs][::K//keep]
                xs = xs[sort_idxs][::K//keep]
                probcoll = probcoll[sort_idxs][::K//keep]
                steers = steers[sort_idxs][::K//keep]

                ys = np.hstack((np.zeros((len(ys), 1)), ys))
                xs = np.hstack((np.zeros((len(xs), 1)), xs))

                # if lines is None:
                axes[1].cla()
                axes[1].plot(0, 0, 'rx', markersize=10)
                # lines = axes[1].plot(np.expand_dims(xs[:,-1], 0), np.expand_dims(ys[:,-1], 0),
                #                      marker='o', linestyle='', markersize=2)
                lines = axes[1].plot(xs.T, ys.T)
                axes[1].plot(xs[0,:], ys[0,:], 'b^', linestyle='', markersize=5)
                axes[1].arrow(0, 0, -2*np.sin(0.5*np.pi * steers[0,0]), 2*np.cos(0.5*np.pi * steers[0,0]), fc='b', ec='b')

                #normalize for color reasons
                # probcoll -= probcoll.min()
                # probcoll /= probcoll.max()
                for l, p in zip(lines, probcoll):
                    l.set_color(cm.viridis(1 - p))
                    l.set_markerfacecolor(cm.viridis(1 - p))

                axes[1].set_xlim(xlim)
                axes[1].set_ylim(ylim)
                axes[1].set_aspect('equal')

                axes[1].set_title('steer {0:.3f}, probcoll in [{1:.2f}, {2:.2f}]'.format(-steers[0, 0], min_probcoll, max_probcoll))

                f.savefig(os.path.join(plot_folder, 'rollout_{0:03d}_t_{1:03d}.png'.format(i, j)),
                          bbox_inches='tight', dpi=200)
                # break
            # break

        plt.close(f)

        tf_sess.close()

def plot_rw_rccar_var001_var016():
    label_params = [
        # ['exp', ('exp_name',)],
        ['policy', ('policy', 'GCGPolicy', 'outputs', 0, 'name')],
        ['H', ('policy', 'H')],
        ['target', ('policy', 'use_target')],
        ['obs_shape', ('alg', 'env', 'params', 'obs_shape')]
    ]

    experiment_groups = [
        ExperimentGroup(os.path.join(DATA_DIR, 'rw_rccar/var{0:03d}'.format(num)),
                        label_params=label_params,
                        plot={
                        })
    for num in [1, 5, 9, 12, 13]]

    mec = MultiExperimentComparison(experiment_groups)

    lengths_list = []
    for exp in mec.list:
        eval_folder = os.path.join(exp.folder, 'eval_itr_0039')
        eval_pkl_fname = os.path.join(eval_folder, 'itr_0039_eval_rollouts.pkl')
        rollouts = mypickle.load(eval_pkl_fname)['rollouts']

        assert (len(rollouts) == 24)

        lengths = [len(r['dones']) for r in rollouts]
        lengths_list.append(lengths)

    f, ax = plt.subplots(1, 1)
    xs = np.vstack((np.r_[0:8.] + 0.,
                    np.r_[0:8.] + 0.1,
                    np.r_[0:8.] + 0.2,)).T.ravel()
    legend_patches = []
    for i, (exp, lengths) in enumerate(zip(mec.list, lengths_list)):
        lengths = np.reshape(lengths, (8, 3))
        width = 0.6 / float(len(lengths_list))
        color = cm.viridis(i / float(len(lengths_list)))
        label = 'median: {0}, {1}'.format(np.median(lengths), exp.plot['label'])

        bp = ax.boxplot(lengths.T, positions=np.arange(len(lengths)) + 1.2 * i * width, widths=width, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
        legend_patches.append(mpatches.Patch(color=color, label=label))
        # ax.plot(xs, lengths, label=exp.plot['label'], linestyle='None', marker='o')
    ax.legend(handles=legend_patches)
    ax.xaxis.set_ticks(np.arange(8))
    ax.xaxis.set_ticklabels(np.arange(8))
    ax.set_xlim((-0.5, 8.5))
    ax.set_xlabel('Start Position Number')
    ax.set_ylabel('Timesteps survived')
    plt.show()

    import IPython; IPython.embed()

if __name__ == '__main__':
    logger.setup(display_name='tmp',
                 log_path='/tmp/log.txt',
                 lvl='debug')

    # plot_test()
    plot_rw_rccar_var001_var016()