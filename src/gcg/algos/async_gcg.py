import os, sys, copy, time
import threading

from gcg.envs.env_utils import create_env
from gcg.policies.gcg_policy import GCGPolicy
from gcg.data.timer import timeit
from gcg.data.logger import logger
from gcg.data import mypickle

from gcg.algos.gcg import GCG


class AsyncGCG(GCG):
    def __init__(self, **kwargs):
        assert (kwargs['save_rollouts'])
        assert (kwargs['save_rollouts_observations'])

        GCG.__init__(self, **kwargs)

        train_args = kwargs['async']['train']
        self._train_save_every_n_steps = train_args['save_every_n_steps']
        self._train_reset_every_n_steps = train_args['reset_every_n_steps']
        self._train_total_steps = train_args.get('total_steps', sys.maxsize)

        infer_args = kwargs['async']['inference']
        self._ssh = infer_args['ssh']
        self._remote_dir = infer_args['remote_dir']
        self._inference_save_every_n_steps = infer_args['save_every_n_steps']
        self._rsync_lock = threading.RLock()

    #####################
    ### Async methods ###
    #####################

    @property
    def _rsync_send_includes(self):
        return ['*_rollouts.pkl']

    @property
    def _rsync_recv_includes(self):
        return ['*_inference_policy_inference.ckpt*']

    def _rsync_thread(self):
        only_exp_dir = os.path.join(*self._save_dir.split('/')[-2:])  # exp_prefix/exp_name

        while True:
            ### rsync for *_rollouts.pkl --> train
            send_include_str = ' '.join(["--include='{0}'".format(include) for include in self._rsync_send_includes])
            send_rsync_cmd = "rsync -az -e 'ssh' {0} --exclude='*' '{1}' {2}:{3}".format(
                send_include_str,
                os.path.join(self._save_dir, ''),
                self._ssh,
                os.path.join(self._remote_dir, only_exp_dir, ''))
            send_retcode = os.system(send_rsync_cmd)
            
            ### rsync for train --> *_inference_policy.ckpt files
            with self._rsync_lock:
                recv_include_str = ' '.join(["--include='{0}'".format(include) for include in self._rsync_recv_includes])
                recv_rsync_cmd = "rsync -az -e 'ssh' {0} --exclude='*' {1}:{2} '{3}'".format(
                    recv_include_str,
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
        inference_itr = 0
        inference_step = 0
        while True:
            fname = self._train_rollouts_file_name(inference_itr)
            if not os.path.exists(fname):
                break

            rollouts = mypickle.load(fname)['rollouts']

            inference_itr += 1
            inference_step += sum([len(r['dones']) for r in rollouts])

        return inference_step

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

    def _train_load_data(self, inference_itr):
        new_inference_itr = self._get_inference_itr()
        if inference_itr < new_inference_itr:
            for i in range(inference_itr, new_inference_itr):
                try:
                    logger.debug('Loading files for itr {0}'.format(i))
                    self._sampler.add_rollouts([self._train_rollouts_file_name(i)])
                    inference_itr = i + 1
                except:
                    logger.debug('Failed to load files for itr {0}'.format(i))

        return inference_itr

    def train(self):
        ### restore where we left off
        init_inference_step = len(self._sampler) # don't count offpolicy
        self._restore_train()
        train_itr = self._get_train_itr()
        train_step = self._get_train_step()
        inference_itr = self._get_inference_itr()

        target_updated = False

        timeit.reset()
        timeit.start('total')
        while True:
            inference_step = len(self._sampler) - init_inference_step
            if inference_step > self._total_steps or train_step > self._train_total_steps:
                break

            if inference_step >= self._learn_after_n_steps:
                ### training step
                train_step += 1
                timeit.start('batch')
                steps, observations, goals, actions, rewards, dones, _ = \
                    self._sampler.sample(self._batch_size)
                timeit.stop('batch')
                timeit.start('train')
                self._policy.train_step(train_step, steps=steps, observations=observations, goals=goals,
                                        actions=actions, rewards=rewards, dones=dones,
                                        use_target=target_updated)
                timeit.stop('train')

                ### update target network
                if train_step > self._update_target_after_n_steps and train_step % self._update_target_every_n_steps == 0:
                    self._policy.update_target()
                    target_updated = True

                ### log
                if train_step % self._log_every_n_steps == 0:
                    logger.info('train itr {0:04d} inference itr {1:04d}'.format(train_itr, inference_itr))
                    logger.record_tabular('Train step', train_step)
                    logger.record_tabular('Inference step', inference_step)
                    self._policy.log()
                    logger.dump_tabular(print_func=logger.info)
                    timeit.stop('total')
                    for line in str(timeit).split('\n'):
                        logger.debug(line)
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
            inference_itr = self._train_load_data(inference_itr)

    def _inference_reset_sampler(self):
        self._sampler.reset()

    def _inference_step(self, inference_step):
        self._sampler.step(inference_step,
                           take_random_actions=(inference_step <= self._onpolicy_after_n_steps),
                           explore=True)
        inference_step += self._sampler.n_envs
        return inference_step

    def inference(self):
        ### restore where we left off
        self._restore_inference()
        inference_itr = self._get_inference_itr()
        inference_step = self._get_inference_step()
        train_itr = self._get_train_itr()

        self._run_rsync()

        train_rollouts = []
        eval_rollouts = []

        self._inference_reset_sampler()

        timeit.reset()
        timeit.start('total')
        while True:
            train_step = self._get_train_step()
            if inference_step > self._total_steps:
                break

            ### sample and add to buffer
            if inference_step > self._sample_after_n_steps:
                timeit.start('sample')
                inference_step = self._inference_step(inference_step)
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
                logger.record_tabular('Train step', train_step)
                logger.record_tabular('Inference step', inference_step)
                self._sampler.log()
                if self._eval_sampler:
                    self._eval_sampler.log(prefix='Eval')
                logger.dump_tabular(print_func=logger.info)
                timeit.stop('total')
                for line in str(timeit).split('\n'):
                    logger.debug(line)
                timeit.reset()
                timeit.start('total')

            ### save rollouts / load model
            train_rollouts += self._sampler.get_recent_paths()
            if inference_step > 0 and inference_step % self._inference_save_every_n_steps == 0:
                self._inference_reset_sampler()

                ### save rollouts
                logger.debug('Saving files for itr {0}'.format(inference_itr))
                self._save_inference(inference_itr, train_rollouts, eval_rollouts)
                inference_itr += 1
                train_rollouts = []
                eval_rollouts = []

                ### load model
                with self._rsync_lock:  # to ensure the ckpt has been fully transferred over
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


def create_async_gcg(params, is_continue, log_fname, AsyncClass=AsyncGCG):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir[:curr_dir.find('src/gcg')], 'data')
    assert (os.path.exists(data_dir))
    save_dir = os.path.join(data_dir, params['exp_name'])
    if os.path.exists(save_dir) and not is_continue:
        print('Save directory {0} exists. You need to explicitly say to continue if you want to start training from where you left off'.format(save_dir))
        sys.exit(0)
    os.makedirs(save_dir, exist_ok=True)
    logger.setup(display_name=params['exp_name'],
                 log_path=os.path.join(save_dir, log_fname),
                 lvl=params['log_level'])

    # copy yaml for posterity
    yaml_path = os.path.join(save_dir, 'params.yaml'.format(params['exp_name']))
    with open(yaml_path, 'w') as f:
        f.write(params['txt'])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU

    env_dict = params['alg'].pop('env')
    env = create_env(env_dict, seed=params['seed'])

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
    algo = AsyncClass(
        save_dir=save_dir,
        env=env,
        env_eval=env_eval,
        policy=policy,
        max_path_length=max_path_length,
        env_dict=env_dict,
        **params['alg']
    )
    return algo


def run_async_gcg_train(params, is_continue, AsyncClass=AsyncGCG):
    algo = create_async_gcg(params, is_continue, log_fname='log_train.txt', AsyncClass=AsyncClass)
    algo.train()


def run_async_gcg_inference(params, is_continue, AsyncClass=AsyncGCG):
    # should only save minimal amount of rollouts in the replay buffer
    params = copy.deepcopy(params)
    params['alg']['offpolicy'] = None
    max_path_length = params['alg']['max_path_length']
    params['alg']['replay_pool_size'] = int(1.5 * max_path_length)
    params['policy']['inference_only'] = True

    algo = create_async_gcg(params, is_continue, log_fname='log_inference.txt', AsyncClass=AsyncClass)
    algo.inference()
