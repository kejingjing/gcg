import os

from rllab.misc.overrides import overrides
import rllab.misc.logger as rllab_logger
from rllab import config

from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.policies.mac_policy import MACPolicy
from sandbox.gkahn.gcg.policies.rccar_mac_policy import RCcarMACPolicy
from sandbox.gkahn.gcg.algos.gcg import GCG
from sandbox.gkahn.gcg.sampler.sampler import Sampler
from sandbox.gkahn.gcg.utils import logger

class RWrccarOffline(GCG):

    def __init__(self, **kwargs):
        self._policy = kwargs['policy']

        self._batch_size = kwargs['batch_size']

        self._sampler = Sampler(
            policy=kwargs['policy'],
            env=kwargs['env'],
            n_envs=1,
            replay_pool_size=int(1e6),
            max_path_length=kwargs['max_path_length'],
            sampling_method='uniform',
            save_rollouts=False,
            save_rollouts_observations=False,
            save_env_infos=False,
            env_str=None,
            replay_pool_params={}
        )

        self._total_steps = int(kwargs['total_steps'])
        self._save_every_n_steps = int(kwargs['save_every_n_steps'])
        self._update_target_every_n_steps = int(kwargs['update_target_every_n_steps'])
        self._log_every_n_steps = int(kwargs['log_every_n_steps'])

        rosbag_folders = kwargs['folders']

    ###############
    ### Restore ###
    ###############

    def _add_offpolicy_rosbags

    ########################
    ### Training methods ###
    ########################

    @overrides
    def train(self):
        ### restore where we left off
        save_itr = self._restore()

        target_updated = False

        self._policy.update_preprocess(self._sampler.statistics)

        for step in range(0, self._total_steps):
                ### training step
                batch = self._sampler.sample(self._batch_size)
                self._policy.train_step(step, *batch, use_target=target_updated)

                ### update target network
                if step > self._update_target_after_n_steps and step % self._update_target_every_n_steps == 0:
                    self._policy.update_target()
                    target_updated = True

                ### log
                if step % self._log_every_n_steps == 0:
                    logger.info('step %.3e' % step)
                    rllab_logger.record_tabular('Step', step)
                    self._sampler.log()
                    self._eval_sampler.log(prefix='Eval')
                    self._policy.log()
                    rllab_logger.dump_tabular(with_prefix=False)

            ### save model
            if step > 0 and step % self._save_every_n_steps == 0:
                logger.info('Saving files for itr {0}'.format(save_itr))
                self._save_train(save_itr)
                save_itr += 1

        self._save_train(save_itr)

def run_rw_rccar_offline(params):
    logger.setup_logger(os.path.join(rllab_logger.get_snapshot_dir(), 'log.txt'),
                        params['log_level'])

    # copy yaml for posterity
    try:
        yaml_path = os.path.join(rllab_logger.get_snapshot_dir(), '{0}.yaml'.format(params['exp_name']))
        with open(yaml_path, 'w') as f:
            f.write(params['txt'])
    except:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU
    config.USE_TF = True

    normalize_env = params['alg'].pop('normalize_env')

    env_str = params['alg'].pop('env')
    env = create_env(env_str, is_normalize=normalize_env, seed=params['seed'])

    env.reset()

    #####################
    ### Create policy ###
    #####################

    policy_class = params['policy']['class']
    PolicyClass = eval(policy_class)
    policy_params = params['policy'][policy_class]

    policy = PolicyClass(
        env_spec=env.spec,
        exploration_strategies=params['alg'].pop('exploration_strategies'),
        **policy_params,
        **params['policy']
    )

    ########################
    ### Create algorithm ###
    ########################

    if 'max_path_length' in params['alg']:
        max_path_length = params['alg'].pop('max_path_length')
    else:
        max_path_length = env.horizon
    algo = RWrccarOffline(
        env=env,
        policy=policy,
        max_path_length=max_path_length,
        batch_size=params['alg']['batch_size']
        **params['alg']['offpolicy_rosbags']
    )
    algo.train()
