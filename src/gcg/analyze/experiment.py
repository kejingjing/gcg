import os, copy
import yaml
import itertools

import pandas
import numpy as np
import matplotlib.pyplot as plt

from gcg.data import mypickle
from gcg.envs.env_utils import create_env
from . import utils

from gcg.policies.gcg_policy import GCGPolicy
from gcg.policies.probcoll_gcg_policy import ProbcollGCGPolicy
from gcg.policies.multisensor_gcg_policy import MultisensorGCGPolicy

############################
### Multiple Experiments ###
############################

class MultiExperimentComparison(object):
    def __init__(self, experiment_groups):
        self._experiment_groups = experiment_groups

    def __getitem__(self, item):
        """
        return experiment with params['exp_name'] == item
        """
        exps = list(itertools.chain(*[eg.experiments for eg in self._experiment_groups]))
        for exp in exps:
            if exp.name == item:
                return exp

        raise Exception('Experiment {0} not found'.format(item))

    ################
    ### Plotting ###
    ################

    def plot_csv(self, keys, save_path=None, **kwargs):
        """
        :param keys: which keys from the csvs do you want to plot
        :param kwargs: save_path, plot_std, avg_window, xlim, ylim
        """
        num_plots = len(keys)

        f, axes = plt.subplots(1, num_plots, figsize=5*np.array([num_plots, 1]))
        if not hasattr(axes, '__iter__'):
            axes = [axes]

        for ax, key in zip(axes[:num_plots], keys):
            self._plot_csv(ax, key, **kwargs)

        if save_path is None:
            plt.show(block=True)
        else:
            f.savefig(save_path,
                      bbox_inches='tight',
                      dpi=kwargs.get('dpi', 100))

    def _plot_csv(self, ax, key, **kwargs):
        avg_window = kwargs.get('avg_window', None)
        plot_std = kwargs.get('plot_std', True)
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)

        for experiment_group in self._experiment_groups:
            csvs = experiment_group.csv

            data_interp = utils.DataAverageInterpolation()
            min_step, max_step = np.inf, -np.inf
            for csv in csvs:
                steps, values = csv['Step'], csv[key]
                if avg_window is not None:
                    steps, values, _ = utils.moving_avg_std(steps, values, window=avg_window)
                data_interp.add_data(steps, values)

                min_step = min(min_step, min(steps))
                max_step = max(max_step, max(steps))

            steps = np.r_[min_step:max_step:50.][1:-1]
            values_mean, values_std = data_interp.eval(steps)

            ax.plot(steps, values_mean, **experiment_group.plot)
            if plot_std:
                plot_params = copy.deepcopy(experiment_group.plot)
                plot_params['label'] = None
                ax.fill_between(steps, values_mean - values_std, values_mean + values_std, alpha=0.4, **plot_params)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.legend(loc='lower right')



########################################
### Single Experiment Multiple Seeds ###
########################################

class ExperimentGroup(object):
    def __init__(self, folder, plot=dict(), clear_obs=False, label_params=None):
        self.plot = plot

        ### load experiments
        fnames = [os.path.join(folder, fname) for fname in os.listdir(folder)]
        all_dirnames =  np.all([os.path.isdir(fname) for fname in fnames])

        if all_dirnames:
            self.experiments = [Experiment(subfolder, plot=plot, clear_obs=clear_obs) for subfolder in fnames]
        else:
            self.experiments = [Experiment(folder, plot=plot, clear_obs=clear_obs)]

        if label_params is not None:
            label = self.get_plot_label(label_params)
            for plot in [self.plot] + [exp.plot for exp in self.experiments]:
                plot['label'] = label


    ####################
    ### Data loading ###
    ####################

    @property
    def params(self):
        return [exp.params for exp in self.experiments]

    @property
    def name(self):
        return [exp.name for exp in self.experiments]

    @property
    def csv(self):
        return [exp.csv for exp in self.experiments]

    @property
    def train_rollouts(self):
        return [exp.train_rollouts for exp in self.experiments]

    @property
    def eval_rollouts(self):
        return [exp.eval_rollouts for exp in self.experiments]

    def get_plot_label(self, label_params):
            """
            :param keys_list: list of keys from self.params to generate label from
            e.g.
                [('policy', 'H'), ('policy', 'get_action', 'K'), ...]
            :return: str
            """
            def nested_get(dct, keys):
                for key in keys:
                    dct = dct[key]
                return dct

            label = ', '.join(['{0}: {1}'.format(k, nested_get(self.params[0], v)) for k, v in label_params])
            return label


#########################
### Single Experiment ###
#########################

class Experiment(object):
    def __init__(self, folder, plot=dict(), clear_obs=False):
        self._folder = folder
        self.plot = plot
        self._clear_obs = clear_obs

        self._internal_params = None
        self._internal_csv = None
        self._internal_train_rollouts = None
        self._internal_eval_rollouts = None

        self.env = None
        self.policy = None

    ########################
    ### Env and policies ###
    ########################

    def create_env(self):
        self.env = create_env(self.params['alg']['env'])

    def create_policy(self):
        assert (self.env is not None)

        policy_class = self.params['policy']['class']
        PolicyClass = eval(policy_class)
        policy_params = self.params['policy'][policy_class]

        self.policy = PolicyClass(
            env_spec=self.env.spec,
            exploration_strategies=None,
            **policy_params,
            **self.params['policy']
        )

    def close_policy(self):
        self.policy.terminate()
        self.policy = None

    #############
    ### Files ###
    #############

    @property
    def _params_file(self):
        return os.path.join(self._folder, 'params.yaml')

    @property
    def _csv_file(self):
        return os.path.join(self._folder, 'log.csv')

    def _train_rollouts_file_name(self, itr):
        return os.path.join(self._folder, 'itr_{0:04d}_train_rollouts.pkl'.format(itr))

    def _eval_rollouts_file_name(self, itr):
        return os.path.join(self._folder, 'itr_{0:04d}_eval_rollouts.pkl'.format(itr))

    ####################
    ### Data loading ###
    ####################

    @property
    def params(self):
        if self._internal_params is None:
            with open(self._params_file, 'r') as f:
                self._internal_params = yaml.load(f)

        return self._internal_params

    @property
    def name(self):
        return self.params['exp_name']

    @property
    def csv(self):
        if self._internal_csv is None:
            self._internal_csv = pandas.read_csv(self._csv_file)

        return self._internal_csv

    @property
    def train_rollouts(self):
        if self._internal_train_rollouts is None:
            self._internal_train_rollouts = self._load_rollouts(self._train_rollouts_file_name)

        return self._internal_train_rollouts

    @property
    def eval_rollouts(self):
        if self._internal_eval_rollouts is None:
            self._internal_eval_rollouts = self._load_rollouts(self._eval_rollouts_file_name)

        return self._internal_eval_rollouts

    def _load_rollouts(self, file_func):
        rollouts_itrs = []
        itr = 0
        while os.path.exists(file_func(itr)):
            rollouts = mypickle.load(file_func(itr))['rollouts']
            if self._clear_obs:
                for r in rollouts:
                    r['observations'] = None
            rollouts_itrs.append(rollouts)
            itr += 1

        return rollouts_itrs