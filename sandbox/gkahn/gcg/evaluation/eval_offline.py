import argparse
import glob
import os
import yaml

import rllab.misc.logger as rllab_logger

from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.utils import mypickle
from sandbox.gkahn.gcg.sampler.replay_pool import ReplayPool

### models
from sandbox.gkahn.gcg.policies.rccar_mac_policy import RCcarMACPolicy
# TODO: import new models

class EvalOffline(object):
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            self._params = yaml.load(f)

        self._folder = os.path.splitext(yaml_path)[0]
        os.makedirs(self._folder, exist_ok=True)
        logger.setup_logger(os.path.join(self._folder, 'log.txt'), self._params['log_level'])

        logger.info('Yaml {0}'.format(yaml_path))
        logger.info('Creating environment')
        self._env = self._create_env()
        logger.info('Creating model')
        self._model = self._create_model()
        logger.info('Loading data')
        self._replay_pool = self._load_data(self._params['offline']['data'])
        if self._params['offline']['checkpoint'] is not None:
            logger.info('Loading checkpoint')
            self._model.restore(self._params['offline']['checkpoint'])

    ###################
    ### Environment ###
    ###################

    def _create_env(self):
        normalize_env = self._params['alg'].pop('normalize_env')
        env_str = self._params['alg'].pop('env')
        return create_env(env_str, is_normalize=normalize_env)

    #############
    ### Model ###
    #############

    def _create_model(self):
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

    ############
    ### Data ###
    ############

    def _load_data(self, folder):
        """
        Loads all .pkl files that can be found recursively from this folder
        """
        assert(os.path.exists(folder))

        rollouts = []
        num_load_success, num_load_fail = 0, 0
        for fname in glob.iglob('{0}/**/*.pkl'.format(folder), recursive=True):
            try:
                rollouts += mypickle.load(fname)['rollouts']
                num_load_success += 1
            except:
                num_load_fail += 1
        logger.info('Files successfully loaded: {0:.2f}%'.format(100. * num_load_success /
                                                                 float(num_load_success + num_load_fail)))

        replay_pool = ReplayPool(
            env_spec=self._env.spec,
            env_horizon=self._env.horizon,
            N=self._model.N,
            gamma=self._model.gamma,
            size=int(1.1 * sum([len(r['dones']) for r in rollouts])),
            obs_history_len=self._model.obs_history_len,
            sampling_method='uniform',
            save_rollouts=False,
            save_rollouts_observations=False,
            save_env_infos=False,
            replay_pool_params={}
        )

        curr_len = 0
        for rollout in rollouts:
            replay_pool.store_rollout(curr_len, rollout)
            curr_len += len(rollout['dones'])

        return replay_pool

    #############
    ### Train ###
    #############

    def _model_checkpoint(self, itr):
        return os.path.join(self._folder, 'itr_{0:04d}_train_policy.ckpt'.format(itr))

    def train(self):
        logger.info('Training model')

        alg_args = self._params['alg']
        total_steps = int(alg_args['total_steps'])
        save_every_n_steps = int(alg_args['save_every_n_steps'])
        update_target_after_n_steps = int(alg_args['update_target_after_n_steps'])
        update_target_every_n_steps = int(alg_args['update_target_every_n_steps'])
        log_every_n_steps = int(alg_args['log_every_n_steps'])
        batch_size = alg_args['batch_size']

        self._model.update_preprocess(self._replay_pool.statistics)

        save_itr = 0
        for step in range(total_steps):
            batch = self._replay_pool.sample(batch_size)
            self._model.train_step(step, *batch, use_target=True)

            ### update target network
            if step > update_target_after_n_steps and step % update_target_every_n_steps == 0:
                self._model.update_target()

            ### log
            if step > 0 and step % log_every_n_steps == 0:
                logger.info('step %.3e' % step)
                rllab_logger.record_tabular('Step', step)
                self._model.log()
                rllab_logger.dump_tabular(with_prefix=False)

            ### save model
            if step > 0 and step % save_every_n_steps == 0:
                logger.info('Saving files for itr {0}'.format(save_itr))
                self._model.save(self._model_checkpoint(save_itr), train=True)
                save_itr += 1

        ### always save the end
        self._model.save(self._model_checkpoint(save_itr), train=True)

    ################
    ### Evaluate ###
    ################

    def evaluate(self):
        logger.info('Evaluating model')

        ### sample from the data, get the outputs
        steps, observations, actions, rewards, values, dones, logprobs = self._replay_pool.sample(10)
        observations = observations[:, :self._model.obs_history_len, :]
        outputs = self._model.get_model_outputs(observations, actions)

        # TODO: do some kind of analysis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml', type=str, help='yaml file with all parameters')
    parser.add_argument('--no_train', action='store_true', help='do not train model on data')
    parser.add_argument('-yamldir', type=str, default=os.path.join(os.path.expanduser('~'),
                                                                   'code/rllab/data/local/offline'))
    args = parser.parse_args()

    yaml_path = os.path.join(args.yamldir, args.yaml)

    eval_offline = EvalOffline(yaml_path)
    if not args.no_train:
        eval_offline.train()
    eval_offline.evaluate()


