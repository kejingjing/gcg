import os

from gcg.analyze.experiment import ExperimentGroup, MultiExperimentComparison

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

if __name__ == '__main__':
    plot_test()