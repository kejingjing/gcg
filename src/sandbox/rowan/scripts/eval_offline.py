import argparse
import glob
import os
import yaml

import numpy as np

from gcg.data import logger
from gcg.envs.env_utils import create_env
from gcg.data import mypickle
from gcg.sampler.replay_pool import ReplayPool

from gcg.policies.gcg_policy import GCGPolicy

from .bnn_plotter import BnnPlotter

class EvalOffline(object):
    def __init__(self, yaml_path, bootstrapping, main_model=None, bootstrap_index=None):
        with open(yaml_path, 'r') as f:
            self._params = yaml.load(f)

        self._bootstrapping = bootstrapping
        self._bootstrap_index = bootstrap_index

        self._folder = os.path.splitext(yaml_path)[0]
        os.makedirs(self._folder, exist_ok=True)
        logger.setup(display_name=params['exp_name'],
                     log_path=os.path.join(self._folder, 'log.txt'),
                     lvl=params['log_level'])

        logger.info('Yaml {0}'.format(yaml_path))

        logger.info('Creating environment')
        if main_model is None:
            self._env = create_env(self._params['alg']['env'])
        else:
            self._env = main_model._env

        logger.info('Creating model')
        self._model = self._create_model()

        logger.info('Loading data')
        self._replay_pool = self._load_data(self._params['offline']['data'])
        self._replay_holdout_pool = self._load_data(self._params['offline']['data_holdout'])

        # TODO: load specific parts
        if self._params['offline']['checkpoint'] is not None:
            logger.info('Loading checkpoint')
            self._model.restore(self._params['offline']['checkpoint'])

        if self._params['offline']['checkpoint_bnn_preprocessing'] is not None:  # TODO: both can't be true, fix.
            logger.info('Loading bnn checkpoint')
            self._model.restore_bnn_preprocessing(self._params['offline']['checkpoint_bnn_preprocessing'])

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

        # TODO: rowan
        if self._bootstrapping:
            num_rollouts = len(rollouts)
            ensemble_indices = np.random.choice(num_rollouts, size=num_rollouts, replace=True)
            rollouts = [rollouts[i] for i in ensemble_indices]

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

        save_itr = 0
        for step in range(total_steps):
            batch = self._replay_pool.sample(batch_size)
            self._model.train_step(step, *batch, use_target=True)

            ### update target network
            if step > update_target_after_n_steps and step % update_target_every_n_steps == 0:
                self._model.update_target()

            ### log
            if step > 0 and step % log_every_n_steps == 0:
                logger.record_tabular('Step', step)
                self._model.log()
                logger.dump_tabular(print_func=logger.info)

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

    @staticmethod
    def clean_rewards(rewards):
        """

        :param rewards: sample_size x action_len. format (negative) 0000010000
        :return: returns rewards                  format (negative) 0000011111
        """
        import numpy as np
        for reward_row in rewards:
            i = np.where(reward_row==-1)[0]
            if i.size > 0:
                i = min(i)
                reward_row[i:] = -1
        return rewards

    @staticmethod
    def evaluate(main_model, all_models, bootstrapping, replay_pool, plotter):
        logger.info('Evaluating model')

        ### sample from the data, get the outputs
        sample_size = 200
        steps, observations, actions, rewards, values, dones, logprobs = replay_pool.sample(sample_size)
        rewards = EvalOffline.clean_rewards(rewards)
        observations = observations[:, :main_model._model.obs_history_len, :]

        outputs = []
        if bootstrapping:
            for model in all_models:
                outputs.append(model._model.get_model_outputs(observations, actions))
        else:
            model = all_models[0]
            num_bnn_samples = 1000
            for _ in range(num_bnn_samples):
                outputs.append(model._model.get_model_outputs(observations, actions))
        outputs = np.asarray(outputs)  # num_bnn_samples x sample_size x action_len

        plotter(outputs, rewards)
        # import IPython; IPython.embed()
        # BnnPlotter.plot_dropout(outputs, rewards)
        # BnnPlotter.plot_predtruth(outputs, rewards)
        # BnnPlotter.plot_hist(outputs, rewards)
        # BnnPlotter.plot_hist_no_time_structure(outputs, rewards)
        # BnnPlotter.plot_roc(outputs, rewards)
        # BnnPlotter.plot_scatter_prob_and_sigma(outputs, rewards)

        # import pickle
        # file_outputs = open("outputs.pkl", 'wb')
        # file_rewards = open("rewards.pkl", 'wb')
        # pickle.dump(outputs, file_outputs)
        # pickle.dump(rewards, file_rewards)

        # file_outputs = open("outputs.pkl", 'rb')
        # file_outputs.seek(0)
        # file_rewards = open("rewards.pkl", 'rb')
        # file_rewards.seek(0)
        # outputs = pickle.load(file_outputs)
        # rewards = pickle.load(file_rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml', type=str, help='yaml file with all parameters')
    parser.add_argument('--no_train', action='store_true', help='do not train model on data')
    parser.add_argument('-yamldir', type=str, default=os.path.join(os.path.expanduser('~'),
                                                                   'code/gcg/data/local/offline'))
    args = parser.parse_args()

    yaml_path = os.path.join(args.yamldir, args.yaml)
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
        bootstrapping = params['policy']['RCcarMACPolicy']['output_graph']['bnn_method'] == 'bootstrap'

    num_bootstraps = 20 if bootstrapping else 1

    main_model = EvalOffline(yaml_path, bootstrapping=False)  # do not train, just a data_loader
    all_models = []
    for i in range(num_bootstraps):
        model = EvalOffline(yaml_path, bootstrapping, main_model, i)
        if not args.no_train:
            model.train()
        all_models.append(model)

    import IPython; IPython.embed()
    EvalOffline.evaluate(main_model, all_models, bootstrapping, main_model._replay_pool, BnnPlotter.plot_dropout)
    EvalOffline.evaluate(main_model, all_models, bootstrapping, main_model._replay_holdout_pool, BnnPlotter.plot_dropout)
