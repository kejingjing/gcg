import os, copy, time
import threading

from rllab.misc.overrides import overrides
import rllab.misc.logger as rllab_logger

from rllab import config

from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.policies.mac_policy import MACPolicy
from sandbox.gkahn.gcg.policies.rccar_mac_policy import RCcarMACPolicy
from sandbox.gkahn.gcg.sampler.sampler import RNNCriticSampler
from sandbox.gkahn.gcg.utils.utils import timeit
from sandbox.gkahn.gcg.utils import logger

from sandbox.gkahn.gcg.algos.gcg import GCG

class AsyncGCG(GCG):

    def __init__(self, **kwargs):
        assert (kwargs['save_rollouts'])
        assert (kwargs['save_rollouts_observations'])

        GCG.__init__(self, **kwargs)

        train_args = kwargs['async']['train']
        self._train_save_every_n_steps = train_args['save_every_n_steps']
        self._train_reset_every_n_steps = train_args['reset_every_n_steps']

        infer_args = kwargs['async']['inference']
        self._ssh = infer_args['ssh']
        self._remote_dir = infer_args['remote_dir']
        self._inference_save_every_n_steps = infer_args['save_every_n_steps']
        self._rsync_lock = threading.RLock()

    #####################
    ### Async methods ###
    #####################

    def _rsync_thread(self):
        only_exp_dir = os.path.join(*self._save_dir.split('/')[-2:]) # exp_prefix/exp_name

        while True:
            ### rsync for *_rollouts.pkl --> train
            send_rsync_cmd = "rsync -az -e 'ssh' --include='{0}' --exclude='*' '{1}' {2}:{3}".format(
                '*_rollouts.pkl',
                os.path.join(self._save_dir, ''),
                self._ssh,
                os.path.join(self._remote_dir, only_exp_dir, ''))
            send_retcode = os.system(send_rsync_cmd)

            ### rsync for train --> *_inference_policy.ckpt files
            with self._rsync_lock:
                recv_rsync_cmd = "rsync -az -e 'ssh' --include='{0}' --exclude='*' {1}:{2} '{3}'".format(
                    '*_inference_policy.ckpt*',
                    self._ssh,
                    os.path.join(self._remote_dir, only_exp_dir, ''),
                    os.path.join(self._save_dir, '')
                )
                recv_ret_code = os.system(recv_rsync_cmd)

            if send_retcode != 0:
                logger.critical('rsync send failed')
            if recv_ret_code != 0:
                logger.critical('rsync receive failed')

            time.sleep(30)

    def _run_rsync(self):
        thread = threading.Thread(target=self._rsync_thread)
        thread.daemon = True
        thread.start()

    ###############
    ### Restore ###
    ###############

    def _get_inference_step(self):
        """ Returns number of steps taken the environment """
        return self._get_inference_itr() * self._save_every_n_steps

    def _get_train_step(self):
        """ Returns number of training steps taken """
        return self._get_train_itr() * self._save_every_n_steps

    def _restore_train(self):
        self._restore_train_rollouts()
        self._restore_train_policy()

    def _restore_inference(self):
        self._restore_inference_policy()

    ########################
    ### Training methods ###
    ########################

    @overrides
    def train(self):
        ### restore where we left off
        self._restore_train()
        train_itr = self._get_train_itr()
        train_step = self._get_train_step()
        inference_itr = self._get_inference_itr()

        target_updated = False

        timeit.reset()
        timeit.start('total')
        while True:
            inference_step = self._get_inference_step()
            if inference_step > self._total_steps:
                break

            if inference_step >= self._learn_after_n_steps:
                ### update preprocess
                if inference_step == self._learn_after_n_steps or inference_step % self._update_preprocess_every_n_steps == 0:
                    self._policy.update_preprocess(self._sampler.statistics)

                ### training step
                train_step += 1
                timeit.start('batch')
                batch = self._sampler.sample(self._batch_size)
                timeit.stop('batch')
                timeit.start('train')
                self._policy.train_step(train_step, *batch, use_target=target_updated)
                timeit.stop('train')

                ### update target network
                if train_step > self._update_target_after_n_steps and train_step % self._update_target_every_n_steps == 0:
                    self._policy.update_target()
                    target_updated = True

                ### log
                if train_step % self._log_every_n_steps == 0:
                    logger.info('train itr {0:04d} inference itr {1:04d}'.format(train_itr, inference_itr))
                    rllab_logger.record_tabular('Train step', train_step)
                    rllab_logger.record_tabular('Inference step', inference_step)
                    self._policy.log()
                    tabular_str = rllab_logger.dump_tabular(with_prefix=False, is_print=False)
                    for l in tabular_str.split('\n'):
                        logger.debug(l)
                    timeit.stop('total')
                    logger.debug('\n'+str(timeit))
                    timeit.reset()
                    timeit.start('total')
            else:
                time.sleep(1)

            ### save model
            if train_step > 0 and train_step % self._train_save_every_n_steps == 0:
                logger.debug('Saving files for itr {0}'.format(train_itr))
                self._save_train(train_itr)
                train_itr += 1

            ### reset model
            if train_step > 0 and self._train_reset_every_n_steps is not None and \
                                    train_step % self._train_reset_every_n_steps == 0:
                logger.debug('Resetting model')
                self._policy.reset_weights()

            ### load data
            new_inference_itr = self._get_inference_itr()
            if inference_itr < new_inference_itr:
                for i in range(inference_itr, new_inference_itr):
                    try:
                        logger.debug('Loading files for itr {0}'.format(i))
                        self._sampler.add_rollouts([self._train_rollouts_file_name(i)])
                        inference_itr = i + 1
                    except:
                        logger.debug('Failed to load files for itr {0}'.format(i))

    def inference(self):
        ### restore where we left off
        self._restore_inference()
        inference_itr = self._get_inference_itr()
        inference_step = self._get_inference_step()
        train_itr = self._get_train_itr()

        self._run_rsync()

        self._sampler.reset()
        if self._eval_sampler:
            self._eval_sampler.reset()
        eval_rollouts = []

        timeit.reset()
        timeit.start('total')
        while True:
            train_step = self._get_train_step()
            if inference_step > self._total_steps:
                break

            ### sample and add to buffer
            if inference_step > self._sample_after_n_steps:
                timeit.start('sample')
                try:
                    self._sampler.step(inference_step,
                                       take_random_actions=(inference_step <= self._learn_after_n_steps or
                                                            inference_step <= self._onpolicy_after_n_steps),
                                       explore=True)
                    inference_step += self._sampler.n_envs
                except Exception as e:
                    logger.warn('Sampler exception {0}'.format(str(e)))
                    trashed_steps = self._sampler.trash_current_rollouts()
                    inference_step -= trashed_steps
                    logger.warn('Trashed {0} steps'.format(trashed_steps))
                    input('Press enter to continue')
                timeit.stop('sample')
            else:
                inference_step += self._sampler.n_envs

            ### sample and DON'T add to buffer (for validation)
            if self._eval_sampler is not None and inference_step > 0 and inference_step % self._eval_every_n_steps == 0:
                timeit.start('eval')
                eval_rollouts_step = []
                eval_step = inference_step
                while len(eval_rollouts_step) == 0:
                    self._eval_sampler.step(eval_step, explore=False)
                    eval_rollouts_step = self._eval_sampler.get_recent_paths()
                    eval_step += 1
                eval_rollouts += eval_rollouts_step
                timeit.stop('eval')

            ### log
            if inference_step % self._log_every_n_steps == 0:
                logger.info('train itr {0:04d} inference itr {1:04d}'.format(train_itr, inference_itr))
                rllab_logger.record_tabular('Train step', train_step)
                rllab_logger.record_tabular('Inference step', inference_step)
                self._sampler.log()
                if self._eval_sampler:
                    self._eval_sampler.log(prefix='Eval')
                tabular_str = rllab_logger.dump_tabular(with_prefix=False, is_print=False)
                for l in tabular_str.split('\n'):
                    logger.debug(l)
                timeit.stop('total')
                logger.debug('\n' + str(timeit))
                timeit.reset()
                timeit.start('total')

            ### save rollouts / load model
            if inference_step > 0 and inference_step % self._inference_save_every_n_steps == 0:
                ### reset to stop rollout
                self._sampler.reset()

                ### save rollouts
                logger.debug('Saving files for itr {0}'.format(inference_itr))
                self._save_inference(inference_itr, self._sampler.get_recent_paths(), eval_rollouts)
                inference_itr += 1
                eval_rollouts = []

                ### load model
                with self._rsync_lock: # to ensure the ckpt has been fully transferred over
                    new_train_itr = self._get_train_itr()
                    if train_itr < new_train_itr:
                        logger.debug('Loading policy for itr {0}'.format(new_train_itr - 1))
                        try:
                            self._policy.restore(self._inference_policy_file_name(new_train_itr - 1), train=False)
                            train_itr = new_train_itr
                        except:
                            logger.debug('Failed to load model for itr {0}'.format(new_train_itr - 1))
                            self._policy.restore(self._inference_policy_file_name(train_itr - 1), train=False)
                            logger.debug('As backup, restored itr {0}'.format(train_itr - 1))

        self._save_inference(inference_itr, self._sampler.get_recent_paths(), eval_rollouts)


def create_async_gcg(params):
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
    if env_eval_str is not None:
        env_eval = create_env(env_eval_str, is_normalize=normalize_env, seed=params['seed'])
    else:
        env_eval = None

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

    max_path_length = params['alg'].pop('max_path_length')
    algo = AsyncGCG(
        env=env,
        env_eval=env_eval,
        policy=policy,
        max_path_length=max_path_length,
        env_str=env_str,
        **params['alg']
    )
    return algo

def run_async_gcg_train(params):
    # setup logger
    logger.setup_logger(os.path.join(rllab_logger.get_snapshot_dir(), 'log_train.txt'),
                        params['log_level'])

    algo = create_async_gcg(params)
    algo.train()

def run_async_gcg_inference(params):
    # setup logger
    logger.setup_logger(os.path.join(rllab_logger.get_snapshot_dir(), 'log_inference.txt'),
                        params['log_level'])

    # should only save minimal amount of rollouts in the replay buffer
    params = copy.deepcopy(params)
    max_path_length = params['alg']['max_path_length']
    params['alg']['replay_pool_size'] = int(1.5 * max_path_length)
    params['policy']['inference_only'] = True

    algo = create_async_gcg(params)
    algo.inference()
