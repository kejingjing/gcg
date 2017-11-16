import os, copy, time
import joblib
import numpy as np
import threading

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab import config

from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.policies.mac_policy import MACPolicy
from sandbox.gkahn.gcg.policies.rccar_mac_policy import RCcarMACPolicy
from sandbox.gkahn.gcg.sampler.sampler import RNNCriticSampler
from sandbox.gkahn.gcg.utils.utils import timeit

from sandbox.gkahn.gcg.algos.gcg import GCG

class AsyncGCG(GCG):

    def __init__(self, **kwargs):
        GCG.__init__(self, **kwargs)

        self._train_ssh = kwargs['async']['train_ssh']
        self._train_data_dir = os.path.join(kwargs['async']['train_data_dir'],
                                            kwargs['exp_prefix'],
                                            kwargs['exp_name'])

    ###########################
    ### Async methods ###
    ###########################

    def _rsync_thread(self):
        while True:
            ### read IP address

            ### rsync for itr_XX_rollouts.pkl --> train
            # rsync -az -e 'ssh' '*_rollouts.pkl' nvidi
            send_rsync_cmd = "rsync -az -e 'ssh' '{0}' {1}:{2}".format(
                os.path.join(self._save_dir, '*_rollouts.pkl'),
                self._train_ssh,
                os.path.join(self._train_data_dir, ''))

            ### rsync for train --> itr_XX.pkl files
            recv_rsync_cmd = "rsync -az -e 'ssh' "

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

            if inference_step >= self._learn_after_n_steps:
                ### update preprocess
                if inference_step == self._learn_after_n_steps or inference_step % self._update_preprocess_every_n_steps == 0:
                    # logger.log('Updating preprocess')
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
                    # logger.log('Updating target network')
                    self._policy.update_target()
                    target_updated = True

                ### log
                if train_step % self._log_every_n_steps == 0:
                    logger.log('train step %.3e' % train_step)
                    logger.log('inference step %.3e' % inference_step)
                    logger.record_tabular('Train step', train_step)
                    logger.record_tabular('Inference step', inference_step)
                    self._policy.log()
                    logger.dump_tabular(with_prefix=False)
                    timeit.stop('total')
                    logger.log('\n'+str(timeit))
                    timeit.reset()
                    timeit.start('total')

            ### save model
            if train_step > 0 and train_step % self._save_every_n_steps == 0:
                logger.log('Saving files')
                self._save_train(train_itr)
                train_itr += 1

            ### load data
            new_inference_itr = self._get_inference_itr()
            if inference_itr < new_inference_itr:
                for i in range(inference_itr, new_inference_itr):
                    self._sampler.add_rollouts(self._train_rollouts_file_name(i))
                inference_itr = new_inference_itr

    def inference(self):
        ### restore where we left off
        self._restore_inference()
        inference_itr = self._get_inference_itr()
        inference_step = self._get_inference_step()
        train_itr = self._get_train_itr()

        self._run_rsync()

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
                self._sampler.step(inference_step,
                                   take_random_actions=(inference_step <= self._learn_after_n_steps or
                                                        inference_step <= self._onpolicy_after_n_steps),
                                   explore=True)
                timeit.stop('sample')

            ### sample and DON'T add to buffer (for validation)
            if inference_step > 0 and inference_step % self._eval_every_n_steps == 0:
                # logger.log('Evaluating')
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
                    logger.log('train step %.3e' % train_step)
                    logger.log('inference step %.3e' % inference_step)
                    logger.record_tabular('Train step', train_step)
                    logger.record_tabular('Inference step', inference_step)
                    self._sampler.log()
                    self._eval_sampler.log(prefix='Eval')
                    logger.dump_tabular(with_prefix=False)
                    timeit.stop('total')
                    logger.log('\n' + str(timeit))
                    timeit.reset()
                    timeit.start('total')

            ### save rollouts
            if inference_step > 0 and inference_step % self._save_every_n_steps == 0:
                logger.log('Saving files')
                self._save_rollouts_file(inference_itr, self._sampler.get_recent_paths())
                self._save_rollouts_file(inference_itr, eval_rollouts, eval=True)
                inference_itr += 1
                eval_rollouts = []

            ### load model
            new_train_itr = self._get_train_itr()
            if train_itr < new_train_itr:
                self._policy.restore(self._inference_policy_file_name(new_train_itr - 1), train=False)
                train_itr = new_train_itr

            ### update step
            inference_step += self._sampler.n_envs

        self._save_rollouts_file(inference_itr, self._sampler.get_recent_paths())
        self._save_rollouts_file(inference_itr, eval_rollouts, eval=True)


def create_async_gcg(params):
    # copy yaml for posterity
    try:
        yaml_path = os.path.join(logger.get_snapshot_dir(), '{0}.yaml'.format(params['exp_name']))
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
    algo = create_async_gcg(params)
    algo.train()

def run_async_gcg_inference(params):
    # should only save minimal amount of rollouts in the replay buffer
    params = copy.deepcopy(params)
    max_path_length = params['alg'].pop('max_path_length')
    params['alg']['replay_pool_size'] = int(1.5 * max_path_length)

    algo = create_async_gcg(params)
    algo.inference()