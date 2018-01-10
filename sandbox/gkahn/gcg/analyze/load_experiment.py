import os
import yaml
import pandas

import rllab.misc.logger as logger

from sandbox.gkahn.gcg.utils import mypickle
from sandbox.gkahn.gcg.envs.env_utils import create_env

class Experiment(object):
    def __init__(self, folder, plot=dict(), create_new_envs=True, clear_obs=False,
                 load_eval_rollouts=True, load_train_rollouts=True):
        """
        :param kwargs: holds random extra properties
        """
        self._folder = folder
        self._create_new_envs = create_new_envs
        self._clear_obs = clear_obs

        ### load data
        # logger.log('AnalyzeRNNCritic: Loading data')
        self.name = os.path.basename(self._folder)
        self.plot = plot
        # logger.log('AnalyzeRNNCritic: params_file: {0}'.format(self._params_file))
        with open(self._params_file, 'r') as f:
            self.params = yaml.load(f)
        # logger.log('AnalyzeRNNCritic: Loading csv')
        try:
            self.progress = pandas.read_csv(self._progress_file)
        except Exception as e:
            logger.log('Could not open csv: {0}'.format(str(e)))
            self.progress = None
        # logger.log('AnalyzeRNNCritic: Loaded csv')

        self.train_rollouts_itrs = self._load_train_rollouts() if load_train_rollouts else None
        self.eval_rollouts_itrs = self._load_eval_rollouts() if load_eval_rollouts else None

        # logger.log('AnalyzeRNNCritic: Loaded all itrs')
        if create_new_envs:
            self.env = create_env(self.params['alg']['env'])
        else:
            self.env = None
        # logger.log('AnalyzeRNNCritic: Created env')
        # logger.log('AnalyzeRNNCritic: Finished loading data')

    #############
    ### Files ###
    #############

    @property
    def _params_file(self):
        yamls = [fname for fname in os.listdir(self._folder) if os.path.splitext(fname)[-1] == '.yaml' and os.path.basename(self._folder) in fname]
        assert(len(yamls) == 1)
        return os.path.join(self._folder, yamls[0])

    @property
    def _progress_file(self):
        return os.path.join(self._folder, 'progress.csv')

    def _train_rollouts_file_name(self, itr):
        return os.path.join(self._folder, 'itr_{0:04d}_train_rollouts.pkl'.format(itr))

    def _eval_rollouts_file_name(self, itr):
        return os.path.join(self._folder, 'itr_{0:04d}_eval_rollouts.pkl'.format(itr))

    ####################
    ### Data loading ###
    ####################

    def _load_train_rollouts(self):
        return self._load_rollouts(self._train_rollouts_file_name)

    def _load_eval_rollouts(self):
        return self._load_rollouts(self._eval_rollouts_file_name)

    def _load_rollouts(self, file_func):
        rollouts_itrs = []
        itr = 0
        while os.path.exists(file_func(itr)):
            rollouts = mypickle.load(self._itr_rollouts_file(itr, eval=eval))['rollouts']
            if self._clear_obs:
                for r in rollouts:
                    r['observations'] = None
            rollouts_itrs.append(rollouts)
            itr += 1

        return rollouts_itrs
