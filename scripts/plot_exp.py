import os, itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from gcg.analyze.experiment import ExperimentGroup, MultiExperimentComparison
from gcg.data.logger import logger

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

def plot_test():
    label_params = [['exp', ('exp_name',)],
                    ['H' ,('policy', 'H')],
                    ['K', ('policy', 'get_action_test', 'random', 'K')]]

    experiment_groups = [
        ExperimentGroup(os.path.join(DATA_DIR, 'sim_rccar/ours'),
                        label_params=label_params,
                        plot={
                            'color': 'r'
                        }),
    ]

    mec = MultiExperimentComparison(experiment_groups)

    mec.plot_csv(['EvalCumRewardMean'],
                 save_path=None,
                 plot_std=True,
                 avg_window=None,
                 xlim=None,
                 ylim=None)

def plot_rw_rccar_var001_var016():
    label_params =[['exp', ('exp_name',)]]

    experiment_groups = [
        ExperimentGroup(os.path.join(DATA_DIR, 'local/rw-rccar/var{0:03d}'.format(i)),
                        label_params=label_params,
                        plot={

                        }) for i in [1]#[1, 3, 5, 7, 9, 11, 13, 15]
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
    for exp in exps:
        exp.create_env()
        exp.create_policy()
        exp.restore_policy()

        # rollouts = list(itertools.chain(*exp.train_rollouts))
        #
        # exp.close_policy()

    import IPython; IPython.embed()

if __name__ == '__main__':
    logger.setup(display_name='tmp',
                 log_path='/tmp/log.txt',
                 lvl='debug')

    # plot_test()
    plot_rw_rccar_var001_var016()