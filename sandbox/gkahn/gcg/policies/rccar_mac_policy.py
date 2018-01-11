import numpy as np
import tensorflow as tf

from rllab.misc import ext

from sandbox.gkahn.gcg.policies.mac_policy import MACPolicy
from sandbox.gkahn.gcg.tf import tf_utils
from sandbox.gkahn.tf.core import xplatform
from sandbox.gkahn.gcg.tf import networks

from sandbox.rocky.tf.spaces.discrete import Discrete

class RCcarMACPolicy(MACPolicy):
    def __init__(self, **kwargs):
        self._speed_weight = kwargs['speed_weight']
        self._is_classification = kwargs['is_classification']
        self._probcoll_strictly_increasing = kwargs['probcoll_strictly_increasing']
        self._coll_weight_pct = kwargs['coll_weight_pct']

        MACPolicy.__init__(self, **kwargs)

        assert(self._H == self._N)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, is_training, num_dp=1):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :return: tf_values: [batch_size, H]
        """
        batch_size = tf.shape(tf_obs_lowd)[0]
        H = tf_actions_ph.get_shape()[1].value
        N = self._N
        assert(self._H == self._N)
        # tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        self._action_graph.update({'output_dim': self._observation_graph['output_dim']})
        action_dim = tf_actions_ph.get_shape()[2].value
        actions = tf.reshape(tf_actions_ph, (-1, action_dim))
        rnn_inputs, _ = networks.fcnn(actions, self._action_graph, is_training=is_training, scope='fcnn_actions',
                                      T=H, global_step_tensor=self.global_step, num_dp=num_dp)
        rnn_inputs = tf.reshape(rnn_inputs, (-1, H, self._action_graph['output_dim']))

        rnn_outputs, _ = networks.rnn(rnn_inputs, self._rnn_graph, initial_state=tf_obs_lowd, num_dp=num_dp)
        rnn_output_dim = rnn_outputs.get_shape()[2].value
        rnn_outputs = tf.reshape(rnn_outputs, (-1, rnn_output_dim))

        self._output_graph.update({'output_dim': 1})
        # individual probcolls
        tf_yhats, _ = networks.fcnn(rnn_outputs, self._output_graph, is_training=is_training, scope='fcnn_yhats',
                                    T=H, global_step_tensor=self.global_step, num_dp=num_dp)
        tf_yhats = tf.reshape(tf_yhats, (-1, H))
        if self._probcoll_strictly_increasing:
            tf_yhats = tf_utils.cumulative_increasing_sum(tf_yhats)
        tf_nstep_rewards = -tf.sigmoid(tf_yhats) if self._is_classification else -tf_yhats
        # future probcolls
        if self._use_target:
            tf_bhats, _ = networks.fcnn(rnn_outputs, self._output_graph, is_training=is_training, scope='fcnn_bhats',
                                        T=H, global_step_tensor=self.global_step, num_dp=num_dp)
            tf_bhats = tf.reshape(tf_bhats, (-1, H))
            tf_nstep_values = -tf.sigmoid(tf_bhats) if self._is_classification else -tf_bhats
        else:
            tf_bhats = None
            tf_nstep_values = tf.zeros((batch_size, H))

        tf_values_list = [(1. / float(h + 1)) * self._graph_calculate_value(h,
                                                                            tf.unstack(tf_nstep_rewards, axis=1),
                                                                            tf.unstack(tf_nstep_values, axis=1))
                          for h in range(H)]
        tf_values = tf.stack(tf_values_list, 1)

        if values_softmax['type'] == 'final':
            tf_values_softmax = tf.zeros([batch_size, N])
            tf_values_softmax[:, -1] = 1.
        elif values_softmax['type'] == 'mean':
            tf_values_softmax = (1. / float(N)) * tf.ones([batch_size, N])
        elif values_softmax['type'] == 'exponential':
            lam = values_softmax['exponential']['lambda']
            lams = (1 - lam) * np.power(lam, np.arange(N - 1))
            lams = np.array(list(lams) + [np.power(lam, N - 1)])
            tf_values_softmax = lams * tf.ones(tf.shape(tf_values))
        else:
            raise NotImplementedError

        assert(tf_values.get_shape()[1].value == H)

        return tf_values, tf_values_softmax, tf_yhats, tf_bhats

    def _graph_get_action(self, tf_obs_ph, get_action_params, scope_select, reuse_select, scope_eval, reuse_eval,
                          tf_episode_timesteps_ph, for_target):
        """
        :param tf_obs_ph: [batch_size, obs_history_len, obs_dim]
        :param get_action_params: how to select actions
        :param scope_select: which scope to evaluate values (double Q-learning)
        :param scope_eval: which scope to select values (double Q-learning)
        :return: tf_get_action [batch_size, action_dim], tf_get_action_value [batch_size]
        """
        ### process to lowd
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_preprocess_select = self._graph_preprocess_placeholders()
            tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess_select, is_training=False)
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_preprocess_eval = self._graph_preprocess_placeholders()
            tf_obs_lowd_eval = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess_eval, is_training=False)

        get_action_type = get_action_params['type']
        if get_action_type == 'random':
            tf_get_action, tf_get_value, tf_get_action_yhats, tf_get_action_bhats, tf_get_action_reset_ops = \
                self._graph_get_action_random(
                    tf_obs_lowd_select, tf_obs_lowd_eval,
                    tf_preprocess_select, tf_preprocess_eval,
                    get_action_params, get_action_type,
                    scope_select, reuse_select,
                    scope_eval, reuse_eval,
                    for_target)
        else:
            raise NotImplementedError

        return tf_get_action, tf_get_value, tf_get_action_yhats, tf_get_action_bhats, tf_get_action_reset_ops

    def _graph_get_action_random(self, tf_obs_lowd_select, tf_obs_lowd_eval, tf_preprocess_select, tf_preprocess_eval,
                                 get_action_params, get_action_type, scope_select, reuse_select, scope_eval, reuse_eval,
                                 for_target):
        H = get_action_params['H']
        assert (H <= self._N)
        add_speed_cost = not for_target

        num_obs = tf.shape(tf_obs_lowd_select)[0]
        action_dim = self._env_spec.action_space.flat_dim
        max_speed = self._env_spec.action_space.high[1]

        ### create actions
        K = get_action_params[get_action_type]['K']
        if isinstance(self._env_spec.action_space, Discrete):
            tf_actions = tf.one_hot(tf.random_uniform([K, H], minval=0, maxval=action_dim, dtype=tf.int32),
                                    depth=action_dim,
                                    axis=2)
        else:
            action_lb = np.expand_dims(self._env_spec.action_space.low, 0)
            action_ub = np.expand_dims(self._env_spec.action_space.high, 0)
            tf_actions = (action_ub - action_lb) * tf.random_uniform([K, H, action_dim]) + action_lb

        ### tile
        tf_actions = tf.tile(tf_actions, (num_obs, 1, 1))
        tf_obs_lowd_repeat_select = tf_utils.repeat_2d(tf_obs_lowd_select, K, 0)
        tf_obs_lowd_repeat_eval = tf_utils.repeat_2d(tf_obs_lowd_eval, K, 0)
        ### inference to get values
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_values_all_select, tf_values_softmax_all_select, _, _ = \
                self._graph_inference(tf_obs_lowd_repeat_select, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_select, is_training=False, num_dp=K)  # [num_obs*k, H]
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_values_all_eval, tf_values_softmax_all_eval, tf_yhats_all_eval, tf_bhats_all_eval = \
                self._graph_inference(tf_obs_lowd_repeat_eval, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_eval, is_training=False, num_dp=K)  # [num_obs*k, H]
        ### get_action based on select (policy)
        tf_values_select = tf.reduce_sum(tf_values_all_select * tf_values_softmax_all_select, reduction_indices=1)  # [num_obs*K]
        if add_speed_cost:
            tf_values_select -= self._speed_weight * tf.reduce_mean(tf.square(tf_actions[:, :, 1] - max_speed),
                                                                    reduction_indices=1)
        tf_values_select = tf.reshape(tf_values_select, (num_obs, K))  # [num_obs, K]
        tf_values_argmax_select = tf.one_hot(tf.argmax(tf_values_select, 1), depth=K)  # [num_obs, K]
        tf_get_action = tf.reduce_sum(
            tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, action_dim)) *
            tf.reshape(tf_actions, (num_obs, K, H, action_dim))[:, :, 0, :],
            reduction_indices=1)  # [num_obs, action_dim]
        ### get_action_value based on eval (target)
        tf_values_eval = tf.reduce_sum(tf_values_all_eval * tf_values_softmax_all_eval, reduction_indices=1)  # [num_obs*K]
        if add_speed_cost:
            tf_values_eval -= self._speed_weight * tf.reduce_mean(tf.square(tf_actions[:, :, 1] - max_speed),
                                                                    reduction_indices=1)
        tf_values_eval = tf.reshape(tf_values_eval, (num_obs, K))  # [num_obs, K]
        tf_get_action_value = tf.reduce_sum(tf_values_argmax_select * tf_values_eval, reduction_indices=1)
        if tf_yhats_all_eval is not None:
            tf_get_action_yhats = tf.reduce_sum(
                tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, H)) * \
                tf.reshape(tf_yhats_all_eval, (num_obs, K, H)),
                1)  # [num_obs, H]
        else:
            tf_get_action_yhats = None # np.nan * tf.ones((num_obs, H))
        if tf_bhats_all_eval is not None:
            tf_get_action_bhats = tf.reduce_sum(
                tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, H)) * \
                tf.reshape(tf_bhats_all_eval, (num_obs, K, H)),
                1)  # [num_obs, H]
        else:
            tf_get_action_bhats = None # np.nan * tf.ones((num_obs, H))

        ### check shapes
        tf.assert_equal(tf.shape(tf_get_action)[0], num_obs)
        tf.assert_equal(tf.shape(tf_get_action_value)[0], num_obs)
        assert(tf_get_action.get_shape()[1].value == action_dim)

        tf_get_action_reset_ops = []

        return tf_get_action, tf_get_action_value, tf_get_action_yhats, tf_get_action_bhats, tf_get_action_reset_ops

    def _graph_cost(self, tf_train_values, tf_train_values_softmax, tf_train_yhats, tf_train_bhats,
                    tf_rewards_ph, tf_dones_ph,
                    tf_target_get_action_values, tf_target_get_action_yhats, tf_target_get_action_bhats):
        tf_dones = tf.cast(tf_dones_ph, tf.int32)
        tf_labels = tf.cast(tf.cumsum(tf_rewards_ph, axis=1) < -0.5, tf.float32)

        ### mask
        if self._clip_cost_target_with_dones:
            lengths = tf.reduce_sum(1 - tf_dones, axis=1) + 1
            all_mask = tf.sequence_mask(
                tf.cast(lengths, tf.int32),
                maxlen=self._H,
                dtype=tf.float32)
            if self._coll_weight_pct is not None:
                factor = tf.reduce_sum(all_mask * tf_labels) / tf.reduce_sum(tf.cast(tf.reduce_sum(lengths), tf.float32))
                coll_weight = (self._coll_weight_pct - self._coll_weight_pct * factor) / (factor - self._coll_weight_pct * factor)
                one_mask = tf.one_hot(
                    tf.cast(lengths - 1, tf.int32),
                    self._H) * (coll_weight - 1.)
                one_mask_coll = one_mask * tf_labels
                mask = tf.cast(all_mask + one_mask_coll, tf.float32)
            else:
                mask = all_mask
        else:
            mask = tf.ones(tf.shape(tf_labels), dtype=tf.float32)
        mask /= tf.reduce_sum(mask)

        ### desired values
        if self._use_target:
            tf_target_get_action_bhats = tf.stop_gradient(tf_target_get_action_bhats)
            if self._is_classification:
                tf_target_get_action_bhats = tf.sigmoid(tf_target_get_action_bhats)

            tf_target_labels = []
            for h in range(self._H):
                tf_target_labels_h = tf.maximum(tf.reduce_max(tf_labels[:, h:], 1),
                                                tf.reduce_max(tf.cast(1 - tf_dones[:, h:], tf.float32) *
                                                              tf_target_get_action_bhats[:, h:], 1))
                tf_target_labels.append(tf_target_labels_h)
            tf_target_labels = tf.stack(tf_target_labels, axis=1)
        else:
            tf_target_labels = None

        ### cost
        control_dependencies = []
        control_dependencies += [tf.assert_greater_equal(tf_labels, 0., name='cost_assert_2')]
        control_dependencies += [tf.assert_less_equal(tf_labels, 1., name='cost_assert_3')]
        with tf.control_dependencies(control_dependencies):
            if self._is_classification:
                cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_train_yhats, labels=tf_labels)
                if tf_target_labels is not None:
                    cross_entropies += tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_train_bhats, labels=tf_target_labels)
                cost = tf.reduce_sum(mask * cross_entropies)
            else:
                mses = tf.square(tf_train_yhats - tf_labels)
                if tf_target_labels is not None:
                    mses += tf.square(tf_train_bhats - tf_target_labels)
                cost = tf.reduce_sum(mask * mses)
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        return cost + weight_decay, cost

    ################
    ### Training ###
    ################

    def train_step(self, step, steps, observations, actions, rewards, values, dones, logprobs, use_target):
        # always True use_target so dones is passed in
        # assert(not self._use_target)
        return MACPolicy.train_step(self, step, steps, observations, actions, rewards, values, dones, logprobs,
                                    use_target=self._use_target) # True: to keep dones to true

