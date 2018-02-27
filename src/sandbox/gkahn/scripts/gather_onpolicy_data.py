import os, argparse, glob
import yaml

import numpy as np

from gcg.data.logger import logger
from gcg.envs.env_utils import create_env
from gcg.sampler.sampler import Sampler
from gcg.data import mypickle

from gcg.policies.gcg_policy import GCGPolicy

class GatherOnpolicyData(object):


    def __init__(self, yaml_path, steps):
        self._steps = steps
        with open(yaml_path, 'r') as f:
            self._params = yaml.load(f)

        logger.setup(display_name=self._params['exp_name'],
                     log_path=os.path.join(self._onpolicy_dir, 'log_onpolicy.txt'),
                     lvl=self._params['log_level'])

        logger.info('Yaml {0}'.format(yaml_path))

        logger.info('')
        logger.info('Creating environment')
        self._env = create_env(self._params['alg']['env'])

        logger.info('')
        logger.info('Creating model')
        self._policy = self._create_policy()
        logger.info('Restoring policy')
        self._restore_policy()

        logger.info('')
        logger.info('Create sampler')
        self._sampler = self._create_sampler()

    def close(self):
        self._policy.terminate()

    #############
    ### Files ###
    #############

    @property
    def _dir(self):
        file_dir = os.path.realpath(os.path.dirname(__file__))
        dir = os.path.join(file_dir[:file_dir.find('gcg/src')], 'gcg/data')
        assert (os.path.exists(dir))
        return dir

    @property
    def _save_dir(self):
        dir = os.path.join(self._dir, self._params['exp_name'])
        os.makedirs(dir, exist_ok=True)
        return dir

    def _train_policy_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_train_policy.ckpt'.format(itr))

    @property
    def _onpolicy_dir(self):
        env_name = self._params['alg']['env'].split('(')[0]
        dir = os.path.join(self._save_dir, '{0}_onpolicy{1}'.format(env_name, self._steps))
        os.makedirs(dir, exist_ok=True)
        return dir

    def _onpolicy_file_name(self, itr):
        return os.path.join(self._onpolicy_dir, 'itr_{0:04d}_onpolicy_rollouts.pkl'.format(itr))

    ############
    ### Init ###
    ############

    def _create_policy(self):
        policy_class = self._params['policy']['class']
        PolicyClass = eval(policy_class)
        policy_params = self._params['policy'][policy_class]

        policy = PolicyClass(
            env_spec=self._env.spec,
            exploration_strategies={},
            **policy_params,
            **self._params['policy']
        )

        return policy

    def _restore_policy(self):
        """
        :return: iteration that it is currently on
        """
        itr = 0
        while len(glob.glob(os.path.splitext(self._train_policy_file_name(itr))[0] + '*')) > 0:
            itr += 1

        if itr > 0:
            logger.info('Loading train policy from iteration {0}...'.format(itr - 1))
            self._policy.restore(self._train_policy_file_name(itr - 1), train=True)
            logger.info('Loaded train policy!')

    def _create_sampler(self):
        return Sampler(policy=self._policy,
                       env=self._env,
                       n_envs=1,
                       replay_pool_size=int(1e5),
                       max_path_length=self._env.horizon,
                       sampling_method='uniform',
                       save_rollouts=True,
                       save_rollouts_observations=True,
                       save_env_infos=True)

    ###########
    ### Run ###
    ###########

    def run(self):
        self._sampler.reset()
        step = 0
        itr = 0
        logger.info('Step {0}'.format(step))
        while step < self._steps:
            self._sampler.step(step, take_random_actions=False)
            step += 1

            if step > 0 and step % 1000 == 0:
                logger.info('Step {0}'.format(step))
                rollouts = self._sampler.get_recent_paths()
                if len(rollouts) > 0:
                    lengths = [len(r['dones']) for r in rollouts]
                    logger.info('Lengths: {0:.1f} +- {1:.1f}'.format(np.mean(lengths), np.std(lengths)))
                    mypickle.dump({'rollouts': rollouts}, self._onpolicy_file_name(itr))
                    itr += 1

        rollouts = self._sampler.get_recent_paths()
        if len(rollouts) > 0:
            logger.info('Step {0}'.format(step))
            lengths = [len(r['dones']) for r in rollouts]
            logger.info('Lengths: {0:.1f} +- {1:.1f}'.format(np.mean(lengths), np.std(lengths)))
            mypickle.dump({'rollouts': rollouts}, self._onpolicy_file_name(itr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('steps', type=int)
    parser.add_argument('--yamls', nargs='+', help='yaml file with all parameters')
    args = parser.parse_args()

    curr_dir = os.path.realpath(os.path.dirname(__file__))
    yaml_dir = os.path.join(curr_dir[:curr_dir.find('gcg/src')], 'gcg/yamls')
    assert (os.path.exists(yaml_dir))
    yaml_paths = [os.path.join(yaml_dir, y) for y in args.yamls]

    for yaml_path in yaml_paths:
        if not os.path.exists(yaml_path):
            print('{0} does not exist'.format(yaml_path))
            raise Exception

    for yaml_path in yaml_paths:
        god = GatherOnpolicyData(yaml_path, args.steps)
        god.run()
        god.close()


