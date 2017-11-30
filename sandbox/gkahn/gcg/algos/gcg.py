import os, glob
import numpy as np

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as rllab_logger
from rllab import config

from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.policies.mac_policy import MACPolicy
from sandbox.gkahn.gcg.policies.rccar_mac_policy import RCcarMACPolicy
from sandbox.gkahn.gcg.sampler.sampler import RNNCriticSampler
from sandbox.gkahn.gcg.utils.utils import timeit
from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.utils import mypickle

class GCG(RLAlgorithm):

    def __init__(self, **kwargs):

        self._policy = kwargs['policy']

        self._batch_size = kwargs['batch_size']
        self._save_rollouts = kwargs['save_rollouts']
        self._save_rollouts_observations = kwargs['save_rollouts_observations']

        self._sampler = RNNCriticSampler(
            policy=kwargs['policy'],
            env=kwargs['env'],
            n_envs=kwargs['n_envs'],
            replay_pool_size=kwargs['replay_pool_size'],
            max_path_length=kwargs['max_path_length'],
            sampling_method=kwargs['replay_pool_sampling'],
            save_rollouts=kwargs['save_rollouts'],
            save_rollouts_observations=kwargs['save_rollouts_observations'],
            save_env_infos=kwargs['save_env_infos'],
            env_str=kwargs['env_str'],
            replay_pool_params=kwargs['replay_pool_params']
        )

        if kwargs['env_eval'] is not None:
            self._eval_sampler = RNNCriticSampler(
                policy=kwargs['policy'],
                env=kwargs['env_eval'],
                n_envs=1,
                replay_pool_size=int(np.ceil(1.5 * kwargs['max_path_length']) + 1),
                max_path_length=kwargs['max_path_length'],
                sampling_method=kwargs['replay_pool_sampling'],
                save_rollouts=True,
                save_rollouts_observations=kwargs.get('save_eval_rollouts_observations', False),
                save_env_infos=kwargs['save_env_infos'],
                replay_pool_params=kwargs['replay_pool_params']
            )
        else:
            self._eval_sampler = None

        if kwargs.get('offpolicy', None) is not None:
            self._add_offpolicy(kwargs['offpolicy'], max_to_add=kwargs['num_offpolicy'])

        alg_args = kwargs
        self._total_steps = int(alg_args['total_steps'])
        self._sample_after_n_steps = int(alg_args['sample_after_n_steps'])
        self._onpolicy_after_n_steps = int(alg_args['onpolicy_after_n_steps'])
        self._learn_after_n_steps = int(alg_args['learn_after_n_steps'])
        self._train_every_n_steps = alg_args['train_every_n_steps']
        self._eval_every_n_steps = int(alg_args['eval_every_n_steps'])
        self._save_every_n_steps = int(alg_args['save_every_n_steps'])
        self._update_target_after_n_steps = int(alg_args['update_target_after_n_steps'])
        self._update_target_every_n_steps = int(alg_args['update_target_every_n_steps'])
        self._update_preprocess_every_n_steps = int(alg_args['update_preprocess_every_n_steps'])
        self._log_every_n_steps = int(alg_args['log_every_n_steps'])
        assert (self._learn_after_n_steps % self._sampler.n_envs == 0)
        if self._train_every_n_steps >= 1:
            assert (int(self._train_every_n_steps) % self._sampler.n_envs == 0)
        assert (self._save_every_n_steps % self._sampler.n_envs == 0)
        assert (self._update_target_every_n_steps % self._sampler.n_envs == 0)
        assert (self._update_preprocess_every_n_steps % self._sampler.n_envs == 0)

    #############
    ### Files ###
    #############

    @property
    def _save_dir(self):
        return rllab_logger.get_snapshot_dir()

    def _train_rollouts_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_train_rollouts.pkl'.format(itr))

    def _eval_rollouts_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_eval_rollouts.pkl'.format(itr))

    def _train_policy_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_train_policy.ckpt'.format(itr))

    def _inference_policy_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_inference_policy.ckpt'.format(itr))

    ############
    ### Save ###
    ############

    def _save_train_rollouts(self, itr, rollouts):
        fname = self._train_rollouts_file_name(itr)
        mypickle.dump({'rollouts': rollouts}, fname)

    def _save_eval_rollouts(self, itr, rollouts):
        fname = self._eval_rollouts_file_name(itr)
        mypickle.dump({'rollouts': rollouts}, fname)

    def _save_train_policy(self, itr):
        self._policy.save(self._train_policy_file_name(itr), train=True)

    def _save_inference_policy(self, itr):
        self._policy.save(self._inference_policy_file_name(itr), train=False)

    def _save_train(self, itr):
        self._save_train_policy(itr)
        self._save_inference_policy(itr)

    def _save_inference(self, itr, train_rollouts, eval_rollouts):
        self._save_train_rollouts(itr, train_rollouts)
        self._save_eval_rollouts(itr, eval_rollouts)

    def _save(self, itr, train_rollouts, eval_rollouts):
        self._save_train(itr)
        self._save_inference(itr, train_rollouts, eval_rollouts)

    ###############
    ### Restore ###
    ###############

    def _add_offpolicy(self, folder, max_to_add):
        assert (os.path.exists(folder))
        logger.info('Loading offpolicy data from {0}'.format(folder))
        rollout_filenames = [os.path.join(folder, fname) for fname in os.listdir(folder) if 'train_rollouts.pkl' in fname]
        self._sampler.add_rollouts(rollout_filenames, max_to_add=max_to_add)
        logger.info('Added {0} samples'.format(len(self._sampler)))

    def _get_train_itr(self):
        train_itr = 0
        while len(glob.glob(self._inference_policy_file_name(train_itr) + '*')) > 0:
            train_itr += 1

        return train_itr

    def _get_inference_itr(self):
        inference_itr = 0
        while os.path.exists(self._train_rollouts_file_name(inference_itr)):
            inference_itr += 1

        return inference_itr

    def _restore_train_rollouts(self):
        """
        :return: iteration that it is currently on
        """
        itr = 0
        rollout_filenames = []
        while True:
            fname = self._train_rollouts_file_name(itr)
            if not os.path.exists(fname):
                break

            rollout_filenames.append(fname)
            itr += 1

        logger.info('Restoring {0} iterations of train rollouts....'.format(itr))
        self._sampler.add_rollouts(rollout_filenames)
        logger.info('Done restoring rollouts!')

    def _restore_train_policy(self):
        """
        :return: iteration that it is currently on
        """
        itr = 0
        while len(glob.glob(self._train_policy_file_name(itr) + '*')) > 0:
            itr += 1

        if itr > 0:
            logger.info('Loading train policy from iteration {0}...'.format(itr - 1))
            self._policy.restore(self._train_policy_file_name(itr - 1), train=True)
            logger.info('Loaded train policy!')

    def _restore_inference_policy(self):
        """
        :return: iteration that it is currently on
        """
        itr = 0
        while len(glob.glob(self._inference_policy_file_name(itr) + '*')) > 0:
            itr += 1

        if itr > 0:
            logger.info('Loading inference policy from iteration {0}...'.format(itr - 1))
            self._policy.restore(self._inference_policy_file_name(itr - 1), train=False)
            logger.info('Loaded inference policy!')

    def _restore(self):
        self._restore_train_rollouts()
        self._restore_train_policy()

        train_itr = self._get_train_itr()
        inference_itr = self._get_inference_itr()
        assert (train_itr == inference_itr,
                'Train itr is {0} but inference itr is {1}'.format(train_itr, inference_itr))
        return train_itr

    ########################
    ### Training methods ###
    ########################

    @overrides
    def train(self):
        ### restore where we left off
        save_itr = self._restore()

        target_updated = False
        eval_rollouts = []

        timeit.reset()
        timeit.start('total')
        for step in range(0, self._total_steps, self._sampler.n_envs):
            ### sample and add to buffer
            if step > self._sample_after_n_steps:
                timeit.start('sample')
                self._sampler.step(step,
                                   take_random_actions=(step <= self._learn_after_n_steps or
                                                        step <= self._onpolicy_after_n_steps),
                                   explore=True)
                timeit.stop('sample')

            ### sample and DON'T add to buffer (for validation)
            if self._eval_sampler is not None and step > 0 and step % self._eval_every_n_steps == 0:
                timeit.start('eval')
                eval_rollouts_step = []
                eval_step = step
                while len(eval_rollouts_step) == 0:
                    self._eval_sampler.step(eval_step, explore=False)
                    eval_rollouts_step = self._eval_sampler.get_recent_paths()
                    eval_step += 1
                eval_rollouts += eval_rollouts_step
                timeit.stop('eval')

            if step >= self._learn_after_n_steps:
                ### update preprocess
                if step == self._learn_after_n_steps or step % self._update_preprocess_every_n_steps == 0:
                    self._policy.update_preprocess(self._sampler.statistics)

                ### training step
                if self._train_every_n_steps >= 1:
                    if step % int(self._train_every_n_steps) == 0:
                        timeit.start('batch')
                        batch = self._sampler.sample(self._batch_size)
                        timeit.stop('batch')
                        timeit.start('train')
                        self._policy.train_step(step, *batch, use_target=target_updated)
                        timeit.stop('train')
                else:
                    for _ in range(int(1. / self._train_every_n_steps)):
                        timeit.start('batch')
                        batch = self._sampler.sample(self._batch_size)
                        timeit.stop('batch')
                        timeit.start('train')
                        self._policy.train_step(step, *batch, use_target=target_updated)
                        timeit.stop('train')

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
                    timeit.stop('total')
                    logger.debug('\n'+str(timeit))
                    timeit.reset()
                    timeit.start('total')

            ### save model
            if step > 0 and step % self._save_every_n_steps == 0:
                logger.info('Saving files for itr {0}'.format(save_itr))
                self._save(save_itr, self._sampler.get_recent_paths(), eval_rollouts)
                save_itr += 1
                eval_rollouts = []

        self._save(save_itr, self._sampler.get_recent_paths(), eval_rollouts)

def run_gcg(params):
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

    env_eval_str = params['alg'].pop('env_eval', env_str)
    env_eval = create_env(env_eval_str, is_normalize=normalize_env, seed=params['seed'])

    env.reset()
    env_eval.reset()

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
    algo = GCG(
        env=env,
        env_eval=env_eval,
        policy=policy,
        max_path_length=max_path_length,
        env_str=env_str,
        **params['alg']
    )
    algo.train()
