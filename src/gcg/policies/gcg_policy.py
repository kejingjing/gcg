import os, itertools
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn.utils.extmath import cartesian

from gcg.envs.spaces.discrete import Discrete
from gcg.policies.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from gcg.policies.exploration_strategies.gaussian_strategy import GaussianStrategy
from gcg.policies.tf import networks
from gcg.policies.tf import tf_utils
from gcg.misc import schedules
from gcg.data.logger import logger


class GCGPolicy(object):
    def __init__(self, **kwargs):
        self._outputs = kwargs['outputs'] 
        self._rew_fn = kwargs['rew_fn']
        
        ### environment
        self._env_spec = kwargs['env_spec']
        self._obs_vec_keys = list(self._env_spec.observation_vec_spec.keys())
        self._action_keys = list(self._env_spec.action_spec.keys())
        self._goal_keys = list(self._env_spec.goal_spec.keys())
        self._output_keys = sorted([output['name'] for output in self._outputs])
        self._obs_im_shape = self._env_spec.observation_im_space.shape

        
        self._obs_im_dim = np.prod(self._obs_im_shape)
        self._obs_vec_dim = len(self._obs_vec_keys)
        self._action_dim = len(self._action_keys)
        self._goal_dim = len(self._goal_keys)
        self._output_dim = len(self._output_keys)
        
        ### model horizons
        self._N = kwargs['N'] # number of returns to use (N-step)
        self._H = kwargs['H'] # action planning horizon for training
        self._gamma = kwargs['gamma'] # reward decay
        self._obs_history_len = kwargs['obs_history_len'] # how many previous observations to use

        ### model architecture
        self._inference_only = kwargs.get('inference_only', False)
        self._image_graph = kwargs['image_graph']
        self._observation_graph = kwargs['observation_graph']
        self._action_graph = kwargs['action_graph']
        self._rnn_graph = kwargs['rnn_graph']
        self._output_graph = kwargs['output_graph']
        ### scopes
        self._image_scope = 'image_scope'
        self._observation_scope = 'observation_scope'
        self._action_scope = 'action_scope'
        self._rnn_scope = 'rnn_scope'
        self._output_scope = 'output_scope'

        ### target network
        self._use_target = kwargs['use_target']
        self._separate_target_params = kwargs['separate_target_params']
        ### training
        self._only_completed_episodes = kwargs['only_completed_episodes']
        self._weight_decay = kwargs['weight_decay']
        self._lr_schedule = schedules.PiecewiseSchedule(**kwargs['lr_schedule'])
        self._grad_clip_norm = kwargs['grad_clip_norm']
        self._gpu_device = kwargs['gpu_device']
        self._gpu_frac = kwargs['gpu_frac']

        ### action selection and exploration
        self._get_action_test = kwargs['get_action_test']
        self._get_action_target = kwargs['get_action_target']
        assert(self._get_action_target['type'] == 'random')
        gaussian_es_params = kwargs['exploration_strategies'].get('GaussianStrategy', None)
        if gaussian_es_params is not None:
            self._gaussian_es = GaussianStrategy(self._env_spec, **gaussian_es_params) if gaussian_es_params else None
        else:
            self._gaussian_es = None
        epsilon_greedy_es_params = kwargs['exploration_strategies'].get('EpsilonGreedyStrategy', None)
        if epsilon_greedy_es_params is not None:
            self._epsilon_greedy_es = EpsilonGreedyStrategy(self._env_spec, **epsilon_greedy_es_params)
        else:
            self._epsilon_greedy_es = None

        ### setup the model
        self._tf_debug = dict()
        self._tf_dict = self._graph_setup()

        ### logging
        self._log_stats = defaultdict(list)

        assert(self._N >= self._H)

    ##################
    ### Properties ###
    ##################

    @property
    def N(self):
        return self._N

    @property
    def gamma(self):
        return self._gamma

    @property
    def session(self):
        return self._tf_dict['sess']

    @property
    def obs_history_len(self):
        return self._obs_history_len

    @property
    def only_completed_episodes(self):
        return self._only_completed_episodes

    ###########################
    ### TF graph operations ###
    ###########################

    @staticmethod
    def create_session_and_graph(gpu_device=None, gpu_frac=None):
        if gpu_device is None:
            gpu_device = 0
        if gpu_frac is None:
            gpu_frac = 0.3

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        tf_graph = tf.Graph()
        if len(str(gpu_device)) > 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            config = tf.ConfigProto(gpu_options=gpu_options,
                                    log_device_placement=False,
                                    allow_soft_placement=True,
                                    inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1)
        else:
            config = tf.ConfigProto(
                device_count={'GPU': 0},
                log_device_placement=False,
                allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
        tf_sess = tf.Session(graph=tf_graph, config=config)
        return tf_sess, tf_graph

    def _graph_input_output_placeholders(self):
        with tf.variable_scope('input_output_placeholders'):
            ### policy inputs
            tf_obs_im_ph = tf.placeholder(tf.uint8, [None, self._obs_history_len, self._obs_im_dim], name='tf_obs_im_ph')
            tf_obs_vec_ph = tf.placeholder(tf.float32, [None, self._obs_history_len, self._obs_vec_dim], name='tf_obs_vec_ph')
            tf_actions_ph = tf.placeholder(tf.float32, [None, self._N + 1, self._action_dim], name='tf_actions_ph')
            tf_dones_ph = tf.placeholder(tf.bool, [None, self._N], name='tf_dones_ph')
            tf_goals_ph = tf.placeholder(tf.float32, [None, self._goal_dim], name='tf_goals_ph')
            ### policy outputs
            tf_rewards_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_rewards_ph')
            ### target inputs
            tf_obs_im_target_ph = tf.placeholder(tf.uint8, [None, self._N + self._obs_history_len - 0, self._obs_im_dim], name='tf_obs_im_target_ph')
            tf_obs_vec_target_ph = tf.placeholder(tf.float32, [None, self._N + self._obs_history_len - 0, self._obs_vec_dim], name='tf_obs_vec_target_ph')
            ### policy exploration
            tf_test_es_ph_dict = defaultdict(None)
            if self._gaussian_es:
                tf_test_es_ph_dict['gaussian'] = tf.placeholder(tf.float32, [None], name='tf_test_gaussian_es')
            if self._epsilon_greedy_es:
                tf_test_es_ph_dict['epsilon_greedy'] = tf.placeholder(tf.float32, [None], name='tf_test_epsilon_greedy_es')
            ### episode timesteps
            tf_episode_timesteps_ph = tf.placeholder(tf.int32, [None], name='tf_episode_timesteps')

        return tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph, tf_dones_ph, tf_goals_ph, tf_rewards_ph,\
               tf_obs_im_target_ph, tf_obs_vec_target_ph, tf_test_es_ph_dict, tf_episode_timesteps_ph

    def _graph_obs_to_lowd(self, tf_obs_im_ph, tf_obs_vec_ph, is_training):
        with tf.name_scope('obs_to_lowd'):
            ### whiten observations
            # TODO: assumes image is an input
            assert (self._obs_im_dim > 0)
            assert (self._image_graph is not None)
            assert (tf_obs_im_ph.dtype == tf.uint8)

            tf_obs_im_whitened = (tf.cast(tf_obs_im_ph, tf.float32) - 128.) / 128.

            ### CNN
            with tf.variable_scope(self._image_scope):
                layer = tf.reshape(tf_obs_im_whitened, [-1, self._obs_history_len] + list(self._obs_im_shape))
                # [batch_size, hist_len, height, width, channels]

                layer = tf.reshape(layer, [-1] + list(self._obs_im_shape))
                # [batch_size * hist_len, height, width, channels]

                layer = networks.convnn(layer, self._image_graph, is_training=is_training, global_step_tensor=self.global_step)
                layer = layers.flatten(layer)
                # pass through cnn to get [batch_size * hist_len, ??]

                layer = tf.reshape(layer, (-1, self._obs_history_len * layer.get_shape()[1].value))
                # [batch_size, hist_len, ??]

            ### FCNN

            with tf.variable_scope(self._observation_scope):
                if tf_obs_vec_ph.get_shape()[1].value > 0:
                    layer = tf.concat([layer, tf.reshape(tf_obs_vec_ph, [-1, self._obs_history_len * self._obs_vec_dim])], axis=1)

                self._observation_graph.update({'batch_size': tf.shape(layer)[0]})
                tf_obs_lowd = networks.fcnn(layer, self._observation_graph, is_training=is_training, global_step_tensor=self.global_step)

        return tf_obs_lowd

    def _graph_inference(self, tf_obs_lowd, inputs, goals, tf_actions_ph, is_training, N=None):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :return: tf_values: [batch_size, H]
        """
        batch_size = tf.shape(tf_obs_lowd)[0]
        H = tf_actions_ph.get_shape()[1].value
        N = self._N if N is None else N
        # tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        with tf.variable_scope(self._action_scope):
            self._action_graph.update({'output_dim': self._observation_graph['output_dim']})
            self._action_graph.update({'batch_size': batch_size})
            rnn_inputs = networks.fcnn(tf_actions_ph, self._action_graph, is_training=is_training,
                                          T=H, global_step_tensor=self.global_step)

        with tf.variable_scope(self._rnn_scope):
            self._rnn_graph['cell_args'].update({'batch_size': batch_size})
            rnn_outputs = networks.rnn(rnn_inputs, self._rnn_graph, initial_state=tf_obs_lowd)

        with tf.variable_scope(self._output_scope):
            self._output_graph.update({'output_dim': self._output_dim})
            self._output_graph.update({'batch_size': batch_size})
            tf_pre_yhats = networks.fcnn(rnn_outputs, self._output_graph, is_training=is_training, scope='fcnn_yhats',
                                                T=H, global_step_tensor=self.global_step)
            tf_pre_bhats = networks.fcnn(rnn_outputs, self._output_graph, is_training=is_training, scope='fcnn_bhats',
                                               T=H, global_step_tensor=self.global_step)

        pre_yhats = OrderedDict()
        for i, key in enumerate(self._output_keys):
            pre_yhats[key] = tf_pre_yhats[:, :, i]
        pre_bhats = OrderedDict()
        for i, key in enumerate(self._output_keys):
            pre_bhats[key] = tf_pre_bhats[:, :, i]

        # Has access to yhats and inputs
        yhats = OrderedDict()
        for output in self._outputs:
            if output['use_yhat']:
                yhats[output['name']] = eval(output['yhat']) 
        # Has access to bhats and inputs
        bhats = OrderedDict()
        for output in self._outputs:
            if output['use_bhat']:
                bhats[output['name']] = eval(output['bhat'])

        values = self._graph_calculate_value(yhats, bhats, goals)
        return values, yhats, bhats

    def _graph_get_action_value_mask(self, mask_args, N):
        if mask_args['type'] == 'final':
            tf_values_mask = tf.zeros([1, N])
            tf_values_mask[:, -1] = 1.
        elif mask_args['type'] == 'mean':
            tf_values_mask = (1. / float(N)) * tf.ones([1, N])
        elif mask_args['type'] == 'exponential':
            lam = mask_args['exponential']['lambda']
            lams = (1 - lam) * np.power(lam, np.arange(N - 1))
            lams = np.array(list(lams) + [np.power(lam, N - 1)])
            tf_values_mask = lams * tf.ones([1, N])
        else:
            raise NotImplementedError
        return tf_values_mask

    def _graph_calculate_value(self, yhats, bhats, goals):
        # Has access to h, yhats, bhats, gammas, goals
        gammas = tf.pow(self._gamma, tf.range(self._H + 1, dtype=tf.float32))
        values = OrderedDict()
        for output in self._outputs:
            if output['use_bhat']:
                outputs_val = []
                for h in range(self._H):
                    outputs_val.append(tf.reshape(eval(output['value']), [-1, 1]))
                value = tf.concat(outputs_val, axis=1) 
                values[output['name']] = tf.reduce_mean(value, axis=1, keep_dims=True)

        return values
    
    def _graph_generate_random_actions(self, K):
        if isinstance(self._env_spec.action_space, Discrete):
            tf_actions = tf.one_hot(tf.random_uniform([K, 1], minval=0, maxval=self._action_dim, dtype=tf.int32),
                                    depth=self._action_dim,
                                    axis=2)
        else:
            action_lb = np.expand_dims(self._env_spec.action_space.low, 0)
            action_ub = np.expand_dims(self._env_spec.action_space.high, 0)
            tf_actions = (action_ub - action_lb) * tf.random_uniform([K, self._action_dim]) + action_lb

        return tf_actions

    def _graph_get_action(self, tf_obs_im_ph, tf_obs_vec_ph, inputs, goals, get_action_params, scope_select, reuse_select,
                          scope_eval, reuse_eval, tf_episode_timesteps_ph):
        """
        :param tf_obs_im_ph: [batch_size, obs_history_len, obs_im_dim]
        :param tf_obs_vec_ph: [batch_size, obs_history_len, obs_vec_dim]
        :param get_action_params: how to select actions
        :param scope_select: which scope to evaluate values (double Q-learning)
        :param scope_eval: which scope to select values (double Q-learning)
        :return: tf_get_action [batch_size, action_dim], tf_get_action_value [batch_size]
        """
        H = get_action_params['H']
        N = self._N
        assert(H <= N)
        get_action_type = get_action_params['type']
        num_obs = tf.shape(tf_obs_im_ph)[0]

        ### create actions
        if get_action_type == 'random':
            K = get_action_params[get_action_type]['K']
            if isinstance(self._env_spec.action_space, Discrete):
                tf_actions = tf.one_hot(tf.random_uniform([K, H], minval=0, maxval=self._action_dim, dtype=tf.int32),
                                        depth=self._action_dim,
                                        axis=2)
            else:
                action_lb = np.expand_dims(self._env_spec.action_space.low, 0)
                action_ub = np.expand_dims(self._env_spec.action_space.high, 0)
                tf_actions = (action_ub - action_lb) * tf.random_uniform([K, H, self._action_dim]) + action_lb
        elif get_action_type == 'lattice':
            assert(isinstance(self._env_spec.action_space, Discrete))
            indices = cartesian([np.arange(self._action_dim)] * H) + np.r_[0:self._action_dim * H:self._action_dim]
            actions = np.zeros((len(indices), self._action_dim * H))
            for i, one_hots in enumerate(indices):
                actions[i, one_hots] = 1
            actions = actions.reshape(len(indices), H, self._action_dim)
            K = len(actions)
            tf_actions = tf.constant(actions, dtype=tf.float32)
        else:
            raise NotImplementedError

        ### process to lowd
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph,
                                                         is_training=False)
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_obs_lowd_eval = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph,
                                                       is_training=False)
        ### tile
        tf_actions = tf.tile(tf_actions, (num_obs, 1, 1))
        tf_obs_lowd_repeat_select = tf_utils.repeat_2d(tf_obs_lowd_select, K, 0)
        tf_obs_lowd_repeat_eval = tf_utils.repeat_2d(tf_obs_lowd_eval, K, 0)
        ### inference to get values
        with tf.variable_scope(scope_select, reuse=reuse_select):
            values_all_select, yhats_all_select, bhats_all_select = \
                self._graph_inference(tf_obs_lowd_repeat_select, inputs, goals, tf_actions,
                                      is_training=False, N=N)  # [num_obs*K, H]
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            values_all_eval, yhats_all_eval, bhats_all_eval = \
                self._graph_inference(tf_obs_lowd_repeat_eval, inputs, goals, tf_actions, is_training=False, N=N)
       
        actions = OrderedDict()
        for i, key in enumerate(self._action_keys):
            actions[key] = tf_actions[:, :, i]
        
        tf_values_select = self._get_action_value(actions, values_all_select, yhats_all_select, bhats_all_select, goals)
        tf_values_eval = self._get_action_value(actions, values_all_eval, yhats_all_eval, bhats_all_eval, goals)
       
        ### get_action based on select (policy)
        tf_values_select = tf.reshape(tf_values_select, (num_obs, K))  # [num_obs, K]
        tf_values_argmax_select = tf.one_hot(tf.argmax(tf_values_select, 1), depth=K)  # [num_obs, K]
        tf_get_action = tf.reduce_sum(
            tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, self._action_dim)) *
            tf.reshape(tf_actions, (num_obs, K, H, self._action_dim))[:, :, 0, :],
            reduction_indices=1)  # [num_obs, action_dim]
        ### get_action_value based on eval (target)
        tf_values_eval = tf.reshape(tf_values_eval, (num_obs, K))  # [num_obs, K]
        tf_get_action_value = tf.reduce_sum(tf_values_argmax_select * tf_values_eval, reduction_indices=1)

        tf_argmax_mask = tf.expand_dims(tf_values_argmax_select, axis=2)
        action_values = OrderedDict()
        for key in values_all_eval.keys():
            action_values[key] = tf.reduce_sum(tf.reshape(values_all_eval[key], (num_obs, K, H)) * tf_argmax_mask, axis=1)

        action_yhats = OrderedDict()
        for key in yhats_all_eval.keys():
            action_yhats[key] = tf.reduce_sum(tf.reshape(yhats_all_eval[key], (num_obs, K, H)) * tf_argmax_mask, axis=1)

        action_bhats = OrderedDict()
        for key in bhats_all_eval.keys():
            action_bhats[key] = tf.reduce_sum(tf.reshape(bhats_all_eval[key], (num_obs, K, H)) * tf_argmax_mask, axis=1)

        ### check shapes
        assert(tf_get_action.get_shape()[1].value == self._action_dim)

        tf_get_action_reset_ops = []

        return tf_get_action, tf_get_action_value, action_values, action_yhats, action_bhats, tf_get_action_reset_ops

    def _get_action_value(self, actions, values, yhats, bhats, goals):
        # can use actions, values, yhats, bhats, goals 
        rew = eval(self._rew_fn)

        value = tf.reduce_sum(rew, axis=1)

        return value

    def _graph_get_action_explore(self, tf_actions, tf_es_ph_dict):
        """
        :param tf_actions: [batch_size, action_dim]
        :param tf_explore_ph: [batch_size]
        :return:
        """
        batch_size = tf.shape(tf_actions)[0]

        ### order below matters (gaussian before epsilon greedy, in case you do both types)
        tf_actions_explore = tf_actions
        if self._gaussian_es:
            tf_explore_ph = tf_es_ph_dict['gaussian']
            tf_actions_explore = tf.clip_by_value(tf_actions_explore + tf.random_normal(tf.shape(tf_actions_explore)) *
                                                  tf.tile(tf.expand_dims(tf_explore_ph, 1), (1, self._action_dim)),
                                                  self._env_spec.action_space.low,
                                                  self._env_spec.action_space.high)
        if self._epsilon_greedy_es:
            tf_explore_ph = tf_es_ph_dict['epsilon_greedy']
            mask = tf.cast(tf.tile(tf.expand_dims(tf.random_uniform([batch_size]) < tf_explore_ph, 1), (1, self._action_dim)), tf.float32)
            tf_actions_explore = (1 - mask) * tf_actions_explore + mask * self._graph_generate_random_actions(batch_size)

        return tf_actions_explore

    def _graph_cost(self, values, yhats, bhats, tf_obs_vec_target_ph, tf_rewards_ph, tf_dones_ph, target_inputs, target_values, target_yhats, target_bhats, N=None):      

        """
        :param tf_rewards_ph: [None, self._N]
        :param tf_dones_ph: [None, self._N]
        :param tf_target_get_action_values: [None, self._N]
        :return: tf_cost, tf_mse
        """
        N = self._N if N is None else N
        assert(tf_rewards_ph.get_shape()[1].value == N)
        assert(tf_dones_ph.get_shape()[1].value == N)

        control_dependencies = []
        costs = []

        batch_size = tf.shape(tf_dones_ph)[0]
        tf_dones = tf.cast(tf_dones_ph, tf.float32)

        ### mask
        lengths = tf.reduce_sum(1 - tf_dones, axis=1)# + 1
        
        control_dependencies.append(tf.assert_greater(lengths, 0., name='length_assert'))
        
        clip_mask = tf.sequence_mask(
            tf.cast(lengths, tf.int32),
            maxlen=self._H,
            dtype=tf.float32)
        clip_mask /= tf.reduce_sum(clip_mask)
        
        no_clip_mask = tf.ones(tf.shape(clip_mask), dtype=tf.float32)
        no_clip_mask /= tf.reduce_sum(no_clip_mask)

        # has access to values, yhats, bhats, target_inputs, target_values, target_yhats, target_bhats, rewards, N
        rewards = tf_rewards_ph
        # TODO not sure if this should be h or n
        gammas = tf.pow(self._gamma, tf.range(self._H + 1, dtype=tf.float32))

        for output in self._outputs:
            if output['clip_with_done']:
                mask = clip_mask
            else:
                mask = no_clip_mask
           
            cost = 0.

            if output['use_yhat']:
                yhat = yhats[output['name']]
                yhat_label = eval(output['yhat_label'])
                yhat_loss = output['yhat_loss']
                yhat_scale = output.get('yhat_scale', 1.0)
                yhat_cost, cost_dep = self._graph_sub_cost(yhat, yhat_label, mask, yhat_loss, yhat_scale)
                cost += yhat_cost
                control_dependencies += cost_dep
            
            if output['use_bhat']:
                bhat = bhats[output['name']]
                bhat_label = self._graph_calculate_bhat_label(output['bhat_label'], target_inputs, target_yhats, target_bhats, target_values, rewards)
#                bhat_label = eval(output['bhat_label'])
                bhat_loss = output['bhat_loss']
                bhat_scale = output.get('bhat_scale', 1.0)
                bhat_cost, cost_dep = self._graph_sub_cost(bhat, bhat_label, mask, bhat_loss, bhat_scale)
                cost += bhat_cost
                control_dependencies += cost_dep

            costs.append(cost)

        with tf.control_dependencies(control_dependencies):
            tf_mse = tf.reduce_sum(costs)

            ### weight decay
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if len(reg_losses) > 0:
                num_trainable_vars = float(np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()]))
                tf_weight_decay = (self._weight_decay / num_trainable_vars) * tf.add_n(reg_losses)
            else:
                tf_weight_decay = 0
            tf_cost = tf_mse + tf_weight_decay

        return tf_cost, tf_mse, costs

    def _graph_sub_cost(self, tf_train_values, tf_labels, mask, loss, scale):
        control_dependencies = []
        if loss == 'mse':
            cost = tf.square(tf_train_values - tf_labels)
        elif loss == 'xentropy':
            tf_labels /= scale
            tf_train_values /= scale
            control_dependencies += [tf.assert_greater_equal(tf_labels, 0., name='cost_assert_2')]
            control_dependencies += [tf.assert_less_equal(tf_labels, 1., name='cost_assert_3')]
            control_dependencies += [tf.assert_greater_equal(tf_train_values, 0., name='cost_assert_4')]
            control_dependencies += [tf.assert_less_equal(tf_train_values, 1., name='cost_assert_5')]
            cost = tf_utils.xentropy(tf_train_values, tf_labels)
        elif loss == 'sin_2':
            cost = tf.square(tf.sin(tf_train_values - tf_labels))
        else:
            raise NotImplementedError
        cost = tf.reduce_sum(cost * mask)
        return cost, control_dependencies

    def _graph_calculate_bhat_label(self, bhat_label_str, target_inputs, target_yhats, target_bhats, target_values, rewards):
        gammas = tf.pow(self._gamma, tf.range(self._H + 1, dtype=tf.float32))
        outputs_val = []
        for h in range(self._H):
            outputs_val.append(tf.reshape(eval(bhat_label_str), [-1, 1]))
        values = tf.concat(outputs_val, axis=1)
        bhat_label = tf.reduce_mean(values, axis=1, keep_dims=True)
        return bhat_label

    def _graph_optimize(self, tf_cost, tf_policy_vars):
        tf_lr_ph = tf.placeholder(tf.float32, (), name="learning_rate")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        num_parameters = 0
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr_ph, epsilon=1e-4)
            gradients = optimizer.compute_gradients(tf_cost, var_list=tf_policy_vars)
            for i, (grad, var) in enumerate(gradients):
                num_parameters += int(np.prod(var.get_shape().as_list()))
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, self._grad_clip_norm), var)
            tf_opt = optimizer.apply_gradients(gradients, global_step=self.global_step)
        logger.debug('Number of parameters: {0:e}'.format(float(num_parameters)))
        return tf_opt, tf_lr_ph

    def _graph_init_vars(self, tf_sess):
        tf_sess.run([tf.global_variables_initializer()])

    def _graph_setup_policy(self, policy_scope, inputs, goals, tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph):
        ### policy
        with tf.variable_scope(policy_scope):
            ### process obs to lowd
            tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec_ph, is_training=True)
            ### create training policy
            values, yhats, bhats = \
                self._graph_inference(tf_obs_lowd, inputs, goals, tf_actions_ph[:, :self._H, :],
                                      is_training=True)

        return values, yhats, bhats

    def _graph_setup_action_selection(self, policy_scope, tf_obs_im_ph, tf_obs_vec_ph, inputs, goals,
                                      tf_episode_timesteps_ph, tf_test_es_ph_dict):
        ### action selection
        tf_get_action, tf_get_action_value, _, _, _, tf_get_action_reset_ops = \
            self._graph_get_action(tf_obs_im_ph, tf_obs_vec_ph, inputs, goals, self._get_action_test,
                                   policy_scope, True, policy_scope, True,
                                   tf_episode_timesteps_ph)
        
        ### exploration strategy and logprob
        tf_get_action_explore = self._graph_get_action_explore(tf_get_action, tf_test_es_ph_dict)

        return tf_get_action, tf_get_action_value, tf_get_action_reset_ops, tf_get_action_explore

    def _graph_setup_target(self, policy_scope, tf_obs_im_target_ph, tf_obs_vec_target_ph,
                            target_inputs, goals, tf_policy_vars):
        ### create target network
        if self._use_target:
            target_scope = 'target' if self._separate_target_params else 'policy'
            ### action selection
            tf_obs_im_target_ph_packed = tf.concat([tf_obs_im_target_ph[:, h - self._obs_history_len:h, :]
                                                    for h in range(self._obs_history_len + 1,
                                                                   self._obs_history_len + self._N + 1)],
                                                   0)
            tf_obs_vec_target_ph_packed = tf.concat([tf_obs_vec_target_ph[:, h - self._obs_history_len:h, :]
                                                     for h in range(self._obs_history_len + 1,
                                                                    self._obs_history_len + self._N + 1)],
                                                    0)
            
            _, _, pre_target_values, pre_target_yhats, pre_target_bhats, _ = self._graph_get_action(tf_obs_im_target_ph_packed, tf_obs_vec_target_ph_packed, target_inputs, goals, self._get_action_target, scope_select=policy_scope, reuse_select=True,
                          scope_eval=target_scope, reuse_eval=(target_scope == policy_scope), tf_episode_timesteps_ph=None) # TODO would need to fill in

            # TODO maybe make self._N + 1
            target_values = OrderedDict()
            for key in pre_target_values.keys():
                target_values[key] = tf.transpose(tf.reshape(pre_target_values[key], (self._N, -1)), (1, 0))

            target_yhats = OrderedDict()
            for key in pre_target_yhats.keys():
                target_yhats[key] = tf.transpose(tf.reshape(pre_target_yhats[key], (self._N, -1, self._N)), (1, 0, 2))

            target_bhats = OrderedDict()
            for key in pre_target_bhats.keys():
                target_bhats[key] = tf.transpose(tf.reshape(pre_target_bhats[key], (self._N, -1, self.N)), (1, 0, 2))

        else:
            target_values = OrderedDict()
            target_yhats = OrderedDict()
            target_bhats = OrderedDict()
        ### update target network
        if self._use_target and self._separate_target_params:
            tf_target_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      scope=target_scope), key=lambda v: v.name)
            tf_update_target_fn = []
            for var, var_target in zip(tf_policy_vars, tf_target_vars):
                assert (var.name.replace(policy_scope, '') == var_target.name.replace(target_scope, ''))
                tf_update_target_fn.append(var_target.assign(var))
            tf_update_target_fn = tf.group(*tf_update_target_fn)
        else:
            tf_target_vars = None
            tf_update_target_fn = None

        return target_values, target_yhats, target_bhats, tf_target_vars, tf_update_target_fn

    def _graph_setup_savers(self, tf_preopt_vars, tf_postopt_vars, inference_only):
        savers_dict = dict()

        savers_dict['inference'] = tf.train.Saver(tf_preopt_vars, max_to_keep=None)

        if not inference_only:
            def filter_policy_vars(must_contain):
                return [v for v in tf_preopt_vars if must_contain in v.name]

            train_vars = dict()
            train_vars['train'] = tf.global_variables()
            train_vars['image'] = filter_policy_vars(self._image_scope)
            train_vars['observation'] = filter_policy_vars(self._observation_scope)
            train_vars['action'] = filter_policy_vars(self._action_scope)
            train_vars['rnn'] = filter_policy_vars(self._rnn_scope)
            train_vars['output'] = filter_policy_vars(self._output_scope)

            for name, vars in train_vars.items():
                savers_dict[name] = tf.train.Saver(vars, max_to_keep=None)

        return savers_dict

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = GCGPolicy.create_session_and_graph(gpu_device=self._gpu_device, gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            ### create input output placeholders
            tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph, tf_dones_ph, tf_goals_ph, tf_rewards_ph,\
            tf_obs_im_target_ph, tf_obs_vec_target_ph, tf_test_es_ph_dict, tf_episode_timesteps_ph = \
                self._graph_input_output_placeholders()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            inputs = OrderedDict()
            for i, key in enumerate(self._obs_vec_keys):
                inputs[key] = tf_obs_vec_ph[:, :, i]
            
            goals = OrderedDict()
            for i, key in enumerate(self._goal_keys):
                goals[key] = tf_goals_ph[:, i]


            ### setup policy
            policy_scope = 'policy'
            values, yhats, bhats = self._graph_setup_policy(policy_scope, inputs, goals, tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph)

            ### get action
            tf_get_action, tf_get_action_value, tf_get_action_reset_ops, tf_get_action_explore = \
                self._graph_setup_action_selection(policy_scope, tf_obs_im_ph, tf_obs_vec_ph, inputs, goals, tf_episode_timesteps_ph, tf_test_es_ph_dict)

            ### get policy variables
            tf_policy_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=policy_scope),
                                    key=lambda v: v.name)
            tf_trainable_policy_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policy_scope),
                                    key=lambda v: v.name)
            tf_preopt_vars = tf.global_variables()

            if not self._inference_only:
                target_inputs = OrderedDict()
                for i, key in enumerate(self._obs_vec_keys):
                    target_inputs[key] = tf_obs_vec_target_ph[:, -self._H:, i]
                
                ### setup target
                target_values, target_yhats, target_bhats, tf_target_vars, tf_update_target_fn = self._graph_setup_target(policy_scope, tf_obs_im_target_ph, tf_obs_vec_target_ph, target_inputs, goals, tf_policy_vars)
                
                ### optimization
                tf_cost, tf_mse, tf_costs = self._graph_cost(values, yhats, bhats, tf_obs_vec_target_ph, tf_rewards_ph, tf_dones_ph, target_inputs, target_values, target_yhats, target_bhats) 
                tf_opt, tf_lr_ph = self._graph_optimize(tf_cost, tf_trainable_policy_vars)
            else:
                tf_costs = []
                tf_target_vars = tf_update_target_fn = tf_cost = tf_mse = tf_opt = tf_lr_ph = None

            ### savers
            tf_postopt_vars = tf.global_variables()
            tf_savers_dict = self._graph_setup_savers(tf_preopt_vars, tf_postopt_vars, self._inference_only)

            ### initialize
            self._graph_init_vars(tf_sess)

        ### what to return
        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'obs_im_ph': tf_obs_im_ph,
            'obs_vec_ph': tf_obs_vec_ph,
            'goals_ph': tf_goals_ph,
            'actions_ph': tf_actions_ph,
            'dones_ph': tf_dones_ph,
            'rewards_ph': tf_rewards_ph,
            'obs_im_target_ph': tf_obs_im_target_ph,
            'obs_vec_target_ph': tf_obs_vec_target_ph,
            'test_es_ph_dict': tf_test_es_ph_dict,
            'episode_timesteps_ph': tf_episode_timesteps_ph,
            'yhats': yhats,
            'bhats': bhats,
            'get_action': tf_get_action,
            'get_action_explore': tf_get_action_explore,
            'get_action_value': tf_get_action_value,
            'get_action_reset_ops': tf_get_action_reset_ops,
            'update_target_fn': tf_update_target_fn,
            'cost': tf_cost,
            'mse': tf_mse,
            'costs': tf_costs,
            'opt': tf_opt,
            'lr_ph': tf_lr_ph,
            'policy_vars': tf_policy_vars,
            'target_vars': tf_target_vars,
            'savers_dict': tf_savers_dict
        }

    ################
    ### Training ###
    ################

    def update_target(self):
        if self._use_target and self._separate_target_params and self._tf_dict['update_target_fn']:
            self._tf_dict['sess'].run(self._tf_dict['update_target_fn'])

    def train_step(self, step, steps, observations, actions, rewards, dones, use_target):
        """
        :param steps: [batch_size, N+1]
        :param observations_im: [batch_size, N+1 + obs_history_len-1, obs_im_dim]
        :param observations_vec: [batch_size, N+1 + obs_history_len-1, obs_vec_dim]
        :param actions: [batch_size, N+1, action_dim]
        :param rewards: [batch_size, N+1]
        :param dones: [batch_size, N+1]
        """
        observations_im, observations_vec = observations
        feed_dict = {
            ### parameters
            self._tf_dict['lr_ph']: self._lr_schedule.value(step),
            ### policy
            self._tf_dict['obs_im_ph']: observations_im[:, :self._obs_history_len, :],
            self._tf_dict['obs_vec_ph']: observations_vec[:, :self._obs_history_len, :],
            self._tf_dict['actions_ph']: actions,
            self._tf_dict['dones_ph']: dones[:, :self._N],
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
            self._tf_dict['obs_vec_target_ph']: observations_vec
        }
        if self._use_target:
            feed_dict[self._tf_dict['obs_im_target_ph']] = observations_im

        cost, mse, costs, _ = self._tf_dict['sess'].run([self._tf_dict['cost'],
                                                        self._tf_dict['mse'],
                                                        self._tf_dict['costs'],
                                                        self._tf_dict['opt']],
                                                        feed_dict=feed_dict)
        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)
        self._log_stats['reg cost'].append(cost - mse)
        for i, output in enumerate(self._outputs):
            self._log_stats['{0} cost'.format(output['name'])].append(costs[i])

    def reset_weights(self):
        tf_sess = self._tf_dict['sess']
        tf_graph = tf_sess.graph
        with tf_sess.as_default(), tf_graph.as_default():
            self._graph_init_vars(tf_sess)

    ######################
    ### Policy methods ###
    ######################

    def get_action(self, step, current_episode_step, observation, goal, explore):
        chosen_actions, chosen_values, action_infos = self.get_actions([step], [current_episode_step] [observation],
                                                                      [goal], explore=explore)
        return chosen_actions[0], chosen_values[0], action_infos[0]

    def get_actions(self, steps, current_episode_steps, observations, goals, explore):
        ds = [{} for _ in steps]
        observations_im, observations_vec = observations
        feed_dict = {
            self._tf_dict['obs_im_ph']: observations_im,
            self._tf_dict['obs_vec_ph']: observations_vec,
            self._tf_dict['goals_ph']: goals,
            self._tf_dict['episode_timesteps_ph']: current_episode_steps
        }
        if explore:
            if self._gaussian_es:
                feed_dict[self._tf_dict['test_es_ph_dict']['gaussian']] = [self._gaussian_es.schedule.value(t) for t in steps]
            if self._epsilon_greedy_es:
                feed_dict[self._tf_dict['test_es_ph_dict']['epsilon_greedy']] = \
                    [self._epsilon_greedy_es.schedule.value(t) for t in steps]

            actions, values = self._tf_dict['sess'].run([self._tf_dict['get_action_explore'],
                                                         self._tf_dict['get_action_value']],
                                                        feed_dict=feed_dict)
        else:
            actions, values = self._tf_dict['sess'].run([self._tf_dict['get_action'],
                                                         self._tf_dict['get_action_value']],
                                                        feed_dict=feed_dict)

        if isinstance(self._env_spec.action_space, Discrete):
            actions = [int(a.argmax()) for a in actions]

        return actions, values, ds
    
    def reset_get_action(self):
        self._tf_dict['sess'].run(self._tf_dict['get_action_reset_ops'])

    def terminate(self):
        self._tf_dict['sess'].close()

    #####################
    ### Model methods ###
    #####################

    def get_model_outputs(self, observations, actions):
        observations_im, observations_vec = observations
        feed_dict = {
            self._tf_dict['obs_im_ph']: observations_im,
            self._tf_dict['obs_vec_ph']: observations_vec,
            self._tf_dict['actions_ph']: actions
        }

        yhats, bhats= self._tf_dict['sess'].run([self._tf_dict['yhats'], self._tf_dict['bhats']], feed_dict=feed_dict)

        return yhats, bhats

    ######################
    ### Saving/loading ###
    ######################

    def _saver_ckpt_name(self, ckpt_name, saver_name):
        name, ext = os.path.splitext(ckpt_name)
        saver_ckpt_name = '{0}_{1}{2}'.format(name, saver_name, ext)
        return saver_ckpt_name

    def save(self, ckpt_name, train=True):
        if train:
            savers_keys = [k for k in self._tf_dict['savers_dict'].keys() if 'inference' not in k]
        else:
            savers_keys = ['inference']

        for saver_name in savers_keys:
            saver = self._tf_dict['savers_dict'][saver_name]
            saver.save(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, saver_name), write_meta_graph=False)

    def restore(self, ckpt_name, train=True, train_restore=('train',)):
        """
        :param: train_restore: 'train', 'image', 'observation', 'action', 'rnn', 'output
        """
        savers_keys = train_restore if train else ['inference']

        for saver_name in savers_keys:
            saver = self._tf_dict['savers_dict'][saver_name]
            saver.restore(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, saver_name))

    ###############
    ### Logging ###
    ###############

    def log(self):
        for k in sorted(self._log_stats.keys()):
            if k == 'Depth':
                logger.record_tabular(k+'Mean', np.mean(self._log_stats[k]))
                logger.record_tabular(k+'Std', np.std(self._log_stats[k]))
            else:
                logger.record_tabular(k, np.mean(self._log_stats[k]))
        self._log_stats.clear()
