from collections import defaultdict
import numpy as np
import tensorflow as tf

from rllab.misc import ext

from avillaflor.gcg.policies.mac_policy import MACPolicy
from avillaflor.gcg.tf import tf_utils
from avillaflor.gcg.tf import networks
from avillaflor.tf.spaces.discrete import Discrete

class RCcarSensorsMACPolicy(MACPolicy):
    def __init__(self, **kwargs):

        #TODO
        self._output_dim = kwargs['output_dim']
        self._output_sensors = kwargs['output_sensors'] 
        self._input_sensors = kwargs['input_sensors']
        self._action_value_terms = kwargs['action_value_terms']
        MACPolicy.__init__(self, **kwargs)

        assert(self._H == self._N)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_extract_from_obs_vec_ph(self, tf_obs_vec_ph, tf_obs_vec_target_ph):
        # TODO Might not be handling target stuff correctly
        # TODO Check again when doing value function stuff
        tf_obs_vec =[]
        tf_obs_vec_target = []
        tf_final_add = []
        tf_final_add_target = []
        tf_goal_vec = []
        final_add_pos = 0
        batch_size = tf.shape(tf_obs_vec_ph)[0]
        for sensor in self._input_sensors:
            use = sensor['use']
            pos = sensor['pos']
            dim = sensor['dim']
            data = tf_obs_vec_ph[:, :, pos:pos+dim]
            data_target = tf_obs_vec_target_ph[:, :, pos:pos+dim]
            if use == 'input':
                tf_obs_vec.append(data)
                tf_obs_vec_target.append(data_target)
            elif use == 'add':
                output_pos = sensor['output_pos']
                data = data[:, -1, :]
                data_target = data_target[:, -1, :]
                if output_pos != final_add_pos:
                    zero_add = tf.zeros((batch_size, output_pos - final_add_pos))
                    tf_final_add.append(zero_add)
                    tf_final_add_target.append(zero_add)
                tf_final_add.append(data)
                tf_final_add_target.append(data_target)
                final_add_pos = output_pos + dim
            elif use == 'goal':
                # TODO consider changing goals
                tf_goal_vec.append(data[:, -1, :])
            else:
                raise NotImplementedError

        if final_add_pos != self._output_dim:
            zero_add = tf.zeros((tf.shape(tf_obs_vec_ph)[0], self._output_dim - final_add_pos))
            tf_final_add.append(zero_add)
            tf_final_add_target.append(zero_add)

        if len(tf_obs_vec) == 0:
            # TODO
            # TODO issue with this being dim 0
            tf_obs_vec = tf.zeros((batch_size, 0))
        else:
            tf_obs_vec = tf.concat(tf_obs_vec, axis=2)
        if len(tf_obs_vec_target) == 0:
            # TODO
            tf_obs_vec_target = tf.zeros((batch_size, 0))
        else:
            tf_obs_vec_target = tf.concat(tf_obs_vec_target, axis=2)
        tf_final_add = tf.concat(tf_final_add, axis=1)
        tf_final_add_target = tf.concat(tf_final_add_target, axis=1)
        tf_goal_vec = tf.concat(tf_goal_vec, axis=1)

        return tf_obs_vec, tf_obs_vec_target, tf_final_add, tf_final_add_target, tf_goal_vec
    
    def _graph_inference(self, tf_obs_lowd, tf_final_add, tf_actions_ph, values_softmax, tf_preprocess, is_training, num_dp=1):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :return: tf_values: [batch_size, H, self._output_dim]
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

        self._output_graph.update({'output_dim': self._output_dim})
        tf_pre_values, _ = networks.fcnn(rnn_outputs, self._output_graph, is_training=is_training, scope='fcnn_values',
                                     T=H, global_step_tensor=self.global_step, num_dp=num_dp)
        tf_pre_values = tf.reshape(tf_pre_values, (-1, H, self._output_dim))

        tf_values = []

        for sensor in self._output_sensors:
            #TODO
            entry = sensor
            dim = entry['dim']
            pos = entry['pos']
            scale = entry.get('scale', 1.0)
            fn = entry.get('fn', None)
            cum_sum = entry.get('cum_sum', None)
            if fn is None:
                activation = tf.identity
            elif fn == 'sigmoid':
                activation = tf.nn.sigmoid
            elif fn == 'tanh':
                activation = tf.nn.tanh
            else:
                raise NotImplementedError
           
            if cum_sum == 'pre':
                specific_vals = tf_pre_values[:, :, pos:pos+dim]
                specific_vals = tf.cumsum(specific_vals, axis=1)
                output_vals = activation(specific_vals) * scale
            elif cum_sum == 'post':
                specific_vals = tf_pre_values[:, :, pos:pos+dim]
                output_vals = activation(specific_vals) * scale
                output_vals = tf.cumsum(output_vals, axis=1)
            elif cum_sum is None:
                specific_vals = tf_pre_values[:, :, pos:pos+dim]
                output_vals = activation(specific_vals) * scale
            else:
                raise NotImplementedError
            tf_values.append(activation(output_vals))

        tf_values = tf.concat(tf_values, axis=2)

        tf_final_add = tf.expand_dims(tf_final_add, axis=1)

        tf_values = tf_values + tf_final_add

        if values_softmax['type'] == 'final':
            tf_values_softmax = tf.zeros([batch_size, N])
            tf_values_softmax[:, -1, :] = 1.
        elif values_softmax['type'] == 'mean':
            tf_values_softmax = (1. / float(N)) * tf.ones([batch_size, N])
        elif values_softmax['type'] == 'exponential':
            lam = values_softmax['exponential']['lambda']
            lams = (1 - lam) * np.power(lam, np.arange(N - 1))
            lams = np.array(list(lams) + [np.power(lam, N - 1)])
            tf_values_softmax = lams * tf.ones([batch_size, N])
        else:
            raise NotImplementedError

        assert(tf_values.get_shape()[1].value == H)

        return tf_values, tf_values_softmax, None, None

    def _graph_get_action(self, tf_obs_im_ph, tf_obs_vec, tf_final_add, tf_goal_vec, 
                          get_action_params, scope_select, reuse_select, scope_eval, reuse_eval,
                          tf_episode_timesteps_ph):
        """
        :param tf_obs_im_ph: [batch_size, obs_history_len, obs_im_dim]
        :param tf_obs_vec: [batch_size, obs_history_len, obs_vec_dim]
        :param tf_final_add: [batch_size, output_dim]
        :param tf_goal_vec: [batch_size, goal_dim]
        :param get_action_params: how to select actions
        :param scope_select: which scope to evaluate values (double Q-learning)
        :param scope_eval: which scope to select values (double Q-learning)
        :return: tf_get_action [batch_size, action_dim], tf_get_action_value [batch_size]
        """
        ### process to lowd
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_preprocess_select = self._graph_preprocess_placeholders()
            tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec, tf_preprocess_select, is_training=False)
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_preprocess_eval = self._graph_preprocess_placeholders()
            tf_obs_lowd_eval = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec, tf_preprocess_eval, is_training=False)

        get_action_type = get_action_params['type']
        if get_action_type == 'random':
            tf_get_action, tf_get_value, tf_get_action_reset_ops = self._graph_get_action_random(
                tf_obs_lowd_select, tf_obs_lowd_eval, tf_final_add, tf_goal_vec,
                tf_preprocess_select, tf_preprocess_eval,
                get_action_params, get_action_type,
                scope_select, reuse_select,
                scope_eval, reuse_eval)
        elif get_action_type == 'cem':
            tf_get_action, tf_get_value, tf_get_action_reset_ops = self._graph_get_action_cem(
                tf_obs_lowd_select, tf_obs_lowd_eval, tf_final_add, tf_goal_vec,
                tf_preprocess_select, tf_preprocess_eval,
                get_action_params, get_action_type, scope_select, reuse_select, scope_eval, reuse_eval,
                tf_episode_timesteps_ph)
        else:
            raise NotImplementedError

        return tf_get_action, tf_get_value, tf_get_action_reset_ops

    def _graph_get_action_random(self, tf_obs_lowd_select, tf_obs_lowd_eval, tf_final_add, tf_goal_vec, tf_preprocess_select, tf_preprocess_eval,
                                 get_action_params, get_action_type, scope_select, reuse_select, scope_eval, reuse_eval):
        H = get_action_params['H']
        assert (H <= self._N)

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
        # For dropout/bootstrap
        tf_actions = tf.tile(tf_actions, (num_obs, 1, 1))
        tf_obs_lowd_repeat_select = tf_utils.repeat_2d(tf_obs_lowd_select, K, 0)
        tf_obs_lowd_repeat_eval = tf_utils.repeat_2d(tf_obs_lowd_eval, K, 0)
        ### inference to get values
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_values_all_select, tf_values_softmax_all_select, _, _ = \
                self._graph_inference(tf_obs_lowd_repeat_select, tf_final_add, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_select, is_training=False, num_dp=K)  # [num_obs*k, H]
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_values_all_eval, tf_values_softmax_all_eval, _, _ = \
                self._graph_inference(tf_obs_lowd_repeat_eval, tf_final_add, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_eval, is_training=False, num_dp=K)  # [num_obs*k, H]
#        if self._is_classification:
#            ### convert pre-activation to post-activation
#            tf_values_all_select = -tf.sigmoid(tf_values_all_select)
#            tf_values_all_eval = -tf.sigmoid(tf_values_all_eval)
#        else:

        # TODO
        tf_values_select = self._get_action_value(tf_actions, tf_values_softmax_all_select, tf_values_all_select, tf_goal_vec)
        tf_values_eval = self._get_action_value(tf_actions, tf_values_softmax_all_eval, tf_values_all_eval, tf_goal_vec)
#        tf_values_all_select = -tf_values_all_select
#        tf_values_all_eval = -tf_values_all_eval

        ### get_action based on select (policy)
#        tf_values_select = tf.reduce_sum(tf_values_all_select * tf_values_softmax_all_select, reduction_indices=(1,))  # [num_obs*K]
        tf_values_select = tf.reshape(tf_values_select, (num_obs, K))  # [num_obs, K]
        tf_values_argmax_select = tf.one_hot(tf.argmax(tf_values_select, 1), depth=K)  # [num_obs, K]
        tf_get_action = tf.reduce_sum(
            tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, action_dim)) *
            tf.reshape(tf_actions, (num_obs, K, H, action_dim))[:, :, 0, :],
            reduction_indices=1)  # [num_obs, action_dim]
        ### get_action_value based on eval (target)
#        tf_values_eval = tf.reduce_sum(tf_values_all_eval * tf_values_softmax_all_eval, reduction_indices=(1,))  # [num_obs*K]
        tf_values_eval = tf.reshape(tf_values_eval, (num_obs, K))  # [num_obs, K]
        tf_get_action_value = tf.reduce_sum(tf_values_argmax_select * tf_values_eval, reduction_indices=1)
        tf_get_action_reset_ops = []

        ### check shapes
        asserts = []
        asserts.append(tf.assert_equal(tf.shape(tf_get_action)[0], num_obs))
        asserts.append(tf.assert_equal(tf.shape(tf_get_action_value)[0], num_obs))
        with tf.control_dependencies(asserts):
            tf_get_action = tf.identity(tf_get_action)
        assert(tf_get_action.get_shape()[1].value == action_dim)

        return tf_get_action, tf_get_action_value, tf_get_action_reset_ops

    def _graph_get_action_cem(self, tf_obs_lowd_select, tf_obs_lowd_eval, tf_final_add, tf_goal_vec, tf_preprocess_select, tf_preprocess_eval,
                              get_action_params, get_action_type, scope_select, reuse_select, scope_eval, reuse_eval,
                              tf_episode_timesteps_ph):
        # TODO update to include the goals and final_add
        H = get_action_params['H']
        assert (H <= self._N)

        num_obs = tf.shape(tf_obs_lowd_select)[0]
        dU = self._env_spec.action_space.flat_dim
        eps = get_action_params['cem']['eps']

        def run_cem(cem_params, distribution):
            init_M = cem_params['init_M']
            M = cem_params['M']
            K = cem_params['K']
            num_additional_iters = cem_params['num_additional_iters']
            Ms = [init_M] + [M] * num_additional_iters
            tf_obs_lowd_repeat_selects = [tf_utils.repeat_2d(tf_obs_lowd_select, init_M, 0)] + \
                                         [tf_utils.repeat_2d(tf_obs_lowd_select, M, 0)] * num_additional_iters

            for M, tf_obs_lowd_repeat_select in zip(Ms, tf_obs_lowd_repeat_selects):
                ### sample from current distribution
                tf_flat_actions_preclip = distribution.sample((M,))
                tf_flat_actions = tf.clip_by_value(
                    tf_flat_actions_preclip,
                    np.array(list(self._env_spec.action_space.low) * H, dtype=np.float32),
                    np.array(list(self._env_spec.action_space.high) * H, dtype=np.float32))
                tf_actions = tf.reshape(tf_flat_actions, (M, H, dU))

                ### eval current distribution costs
                with tf.variable_scope(scope_select, reuse=reuse_select):
                    tf_values_all_select, tf_values_softmax_all_select, _, _ = \
                        self._graph_inference(tf_obs_lowd_repeat_select, tf_final_add, tf_actions,
                                              get_action_params['values_softmax'],
                                              tf_preprocess_select, is_training=False, num_dp=M)  # [num_obs*k, H]

#                if self._is_classification:
#                    tf_values_all_select = -tf.sigmoid(tf_values_all_select) # convert pre-activation to post-activation
#                else:
                tf_values_all_select = -tf_values_all_select

                tf_values_select = tf.reduce_sum(tf_values_all_select * tf_values_softmax_all_select,
                                                 reduction_indices=1)  # [num_obs*K] # TODO: if variable speed, need to multiple by kinetic energy

                ### get top k
                _, top_indices = tf.nn.top_k(tf_values_select, k=K)
                top_controls = tf.gather(tf_flat_actions, indices=top_indices)

                ### set new distribution based on top k
                mean = tf.reduce_mean(top_controls, axis=0)
                covar = tf.matmul(tf.transpose(top_controls), top_controls) / float(K)
                sigma = covar + eps * tf.eye(H * dU)

                distribution = tf.contrib.distributions.MultivariateNormalFullCovariance(
                    loc=mean,
                    covariance_matrix=sigma
                )

            return tf_values_select, tf_actions

        control_dependencies = []
        control_dependencies += [tf.assert_equal(num_obs, 1)]
        control_dependencies += [tf.assert_equal(tf.shape(tf_episode_timesteps_ph)[0], 1)]
        with tf.control_dependencies(control_dependencies):
            with tf.variable_scope('cem_warm_start', reuse=False):
                mu = tf.get_variable('mu', [dU * H], trainable=False)
            tf_get_action_reset_ops = [mu.initializer]

            control_lower = np.array(self._env_spec.action_space.low.tolist() * H, dtype=np.float32)
            control_upper = np.array(self._env_spec.action_space.high.tolist() * H, dtype=np.float32)
            control_std = np.square(control_upper - control_lower) / 12.0
            init_distribution = tf.contrib.distributions.Uniform(control_lower, control_upper)
            gauss_distribution = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=control_std)

            tf_values_select, tf_actions = tf.cond(tf.greater(tf_episode_timesteps_ph[0], 0),
                                   lambda: run_cem(get_action_params['cem']['warm_start'], gauss_distribution),
                                   lambda: run_cem(get_action_params['cem'], init_distribution))

            ### get action from best of last batch
            tf_get_action_index = tf.cast(tf.argmax(tf_values_select, axis=0), tf.int32)
            tf_get_action_seq = tf_actions[tf_get_action_index]

            ### update mu for warm starting
            tf_get_action_seq_flat_end = tf.reshape(tf_get_action_seq[1:], (dU * (H - 1), ))
            next_mean = tf.concat([tf_get_action_seq_flat_end, tf_get_action_seq_flat_end[-dU:]], axis=0)
            update_mean = tf.assign(mu, next_mean)
            with tf.control_dependencies([update_mean]):
                tf_get_action = tf_get_action_seq[0]

            ### get_action_value based on eval (target)
            with tf.variable_scope(scope_eval, reuse=reuse_eval):
                tf_actions = tf.expand_dims(tf_get_action_seq, 0)
                tf_values_all_eval, tf_values_softmax_all_eval, _, _ = \
                    self._graph_inference(tf_obs_lowd_eval, tf_final_add, tf_actions, get_action_params['values_softmax'],
                                          tf_preprocess_eval, is_training=False)  # [num_obs*k, H]

#                if self._is_classification:
#                    tf_values_all_eval = -tf.sigmoid(tf_values_all_eval) # convert pre-activation to post-activation
#                else:
                tf_values_all_eval = -tf_values_all_eval

                tf_values_eval = tf.reduce_sum(tf_values_all_eval * tf_values_softmax_all_eval,
                                               reduction_indices=1)  # [num_obs*K] # TODO: if variable speed, need to multiple by kinetic energy
                tf_get_action_value = tf_values_eval

            return tf_get_action, tf_get_action_value, tf_get_action_reset_ops

    def _get_action_value(self, tf_actions, tf_mask, tf_graph_output_values, tf_goal_vec):
        tf_mask = tf.expand_dims(tf_mask, axis=2)
        value = 0.
        for action_value_term in self._action_value_terms:
            ty = action_value_term.get('type')
            alpha = action_value_term.get('alpha', 1.0)
            fn = action_value_term.get('fn', None)
            if fn == 'square':
                tf_fn = tf.square
            elif fn == 'cos':
                tf_fn = lambda x: (tf.cos(x) + 1.) / 2.
            elif fn == 'shift':
                tf_fn = lambda x: (x + 1.) / 2.
            elif fn is None:
                tf_fn = tf.identity
            else:
                raise NotImplementedError
            if ty == 'probnocoll':
                fn_value = tf_fn(1. - tf_graph_output_values[:, :, 0:1])
                value += alpha * tf.reduce_sum(tf_mask * fn_value, axis=[1, 2]) 
            else:
                dim = action_value_term.get('dim')
                pos = action_value_term.get('pos')
                multiply = action_value_term.get('multiply', None)
                if multiply == 'probcoll':
                    multiplier = tf_graph_output_values[:, :, 0:1]
                elif multiply == 'probnocoll':
                    multiplier = 1. - tf_graph_output_values[:, :, 0:1]
                elif multiply is None:
                    multiplier = 1.
                else:
                    raise NotImplementedError
                if ty == 'goal':
                    goal_pos = action_value_term.get('goal_pos')
                    goal = tf.expand_dims(tf_goal_vec[:, goal_pos:goal_pos+dim], axis=1)
                    val = tf_graph_output_values[:, :, pos:pos+dim]
                    fn_value = tf_fn(goal - val) * multiplier
                    value += alpha * tf.reduce_sum(tf_mask * fn_value, axis=[1, 2])
                elif ty == 'u_cost':
                    fn_value = tf_fn(tf_actions[:, :, pos:pos+dim]) * multiplier
                    value += alpha * tf.reduce_sum(tf_mask * fn_value, axis=[1, 2]) 
        return value

    def _graph_cost(self, tf_train_values, tf_rewards_ph, tf_dones_ph, tf_obs_im_target_ph,
                    tf_obs_vec_target_ph, tf_target_get_action_values):
        tf_dones = tf.cast(tf_dones_ph, tf.int32)
        ### mask
#        if self._clip_cost_target_with_dones:
#            lengths = tf.reduce_sum(1 - tf_dones, axis=1) + 1
#            mask = tf.sequence_mask(
#                tf.cast(lengths, tf.int32),
#                maxlen=self._H,
#                dtype=tf.float32)
##            mask = tf.tile(tf.expand_dims(mask, 1), [1, self._output_dim, 1])
#            mask = tf.expand_dims(mask, 2)
#        else:
#            mask = tf.ones(tf.shape(tf_train_values), dtype=tf.float32)
#        mask /= tf.reduce_sum(mask)

        lengths = tf.reduce_sum(1 - tf_dones, axis=1) + 1
        clip_mask = tf.sequence_mask(
            tf.cast(lengths, tf.int32),
            maxlen=self._H,
            dtype=tf.float32)
#            mask = tf.tile(tf.expand_dims(mask, 1), [1, self._output_dim, 1])
        clip_mask = tf.expand_dims(clip_mask, 2)
        clip_mask /= tf.reduce_sum(clip_mask)
        
        no_clip_mask = tf.ones(tf.shape(clip_mask), dtype=tf.float32)
#        no_clip_mask = tf.ones(tf.shape(tf_train_values), dtype=tf.float32)
        no_clip_mask /= tf.reduce_sum(no_clip_mask)

        control_dependencies = []
        costs = []

        for sensor in self._output_sensors:
            # TODO
            entry = sensor
            dim = entry['dim']
            goal_pos = entry['goal_pos']
            goal_type = entry['goal']
            clip_with_done = entry.get('clip_with_done', True)
            name = entry['name']
            if clip_with_done:
                mask = clip_mask
            else:
                mask = no_clip_mask
            # TODO
            if goal_type == 'reward':
                goal = tf_rewards_ph
            elif goal_type == 'obs_vec':
                goal = tf_obs_vec_target_ph[:, -self._H:, goal_pos:goal_pos+dim] 
            pos = entry['pos']
            loss = entry.get('loss', 'mse')
            scale = entry.get('scale', 1.)
#            if sensor == 'probcoll':
            if pos == 0:
#                specific_rew = tf.slice(tf_rewards_ph, [0, goal_pos], [tf.shape(tf_rewards_ph)[0], dim])
#                tf_val_labels = tf.cast(tf.cumsum(specific_rew, axis=1) < -0.5, tf.float32)
                tf_val_labels = tf.cast(tf.cumsum(tf_rewards_ph, axis=1) < -0.5, tf.float32)
                #TODO
                specific_vals = tf_train_values[:, :, pos:pos+dim]
#                specific_vals = tf.slice(tf_train_values, [0, 0, pos], [tf.shape(tf_train_values)[0], tf.shape(tf_train_values)[1], dim])
                tf_val_labels = tf.reshape(tf_val_labels, tf.shape(specific_vals))
                #TODO figure out how to get tf_labels
                tf_labels = tf_val_labels
            
                ### desired values
                # assert(not self._use_target)
                coll_cost, coll_dep = self._graph_sub_gt_cost(specific_vals, tf_labels, loss, scale)
                cost = tf.reduce_sum(coll_cost * mask)
                costs.append(cost)
                control_dependencies += coll_dep
            else:
                # TODO
                specific_vals = tf_train_values[:, :, pos:pos+dim]
#                specific_vals = tf.slice(tf_train_values, [0, 0, pos], [tf.shape(tf_train_values)[0], tf.shape(tf_train_values)[1], dim])
                tf_labels = goal
                coll_cost, coll_dep = self._graph_sub_gt_cost(specific_vals, tf_labels, loss, scale)
                cost = tf.reduce_sum(coll_cost * mask)
                costs.append(cost)
                control_dependencies += coll_dep
#        if self._use_target:
#            target_labels = tf.stop_gradient(-tf_target_get_action_values)
#            control_dependencies += [tf.assert_greater_equal(target_labels, 0., name='cost_assert_0')]
#            control_dependencies += [tf.assert_less_equal(target_labels, 1., name='cost_assert_1')]
#            with tf.control_dependencies(control_dependencies):
#                tf_labels = tf.cast(tf_dones_ph, tf.float32) * tf_labels + \
#                            (1 - tf.cast(tf_dones_ph, tf.float32)) * tf.maximum(tf_labels, target_labels)

        ### cost
#        control_dependencies += [tf.assert_greater_equal(tf_labels, 0., name='cost_assert_2')]
#        control_dependencies += [tf.assert_less_equal(tf_labels, 1., name='cost_assert_3')]
        with tf.control_dependencies(control_dependencies):
#            if self._is_classification:
#                cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_train_values, labels=tf_labels)
#                cost = tf.reduce_sum(mask * cross_entropies)
#            else:
#            mses = tf.square(tf_train_values - tf_labels)
#            mses = tf.reduce_sum(costs, axis=0)
#            cost = tf.reduce_sum(mses)
            cost = tf.reduce_sum(costs)
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        return cost + weight_decay, cost, costs

    def _graph_sub_gt_cost(self, tf_train_values, tf_labels, loss, scale):
        #TODO
        control_dependencies = []
        if loss == 'mse':
            cost = tf.reduce_sum(tf.square(tf_train_values - tf_labels), axis=2, keep_dims=True) 
        elif loss == 'xentropy':
            tf_labels /= scale
            tf_train_values /= scale
            control_dependencies += [tf.assert_greater_equal(tf_labels, 0., name='cost_assert_2')]
            control_dependencies += [tf.assert_less_equal(tf_labels, 1., name='cost_assert_3')]
            control_dependencies += [tf.assert_greater_equal(tf_train_values, 0., name='cost_assert_4')]
            control_dependencies += [tf.assert_less_equal(tf_train_values, 1., name='cost_assert_5')]
            cost = tf_utils.xentropy(tf_train_values, tf_labels)
        elif loss == 'sin_2':
            cost = tf.reduce_sum(tf.square(tf.sin(tf_train_values - tf_labels)), axis=2, keep_dims=True)
        else:
            raise NotImplementedError
        return cost, control_dependencies

    def _graph_sub_target_cost(self, tf_train_values, tf_rewards_ph, tf_dones_ph, tf_obs_im_target_ph,
                               tf_obs_vec_target_ph, tf_target_get_action_values, loss):
        if self._use_target:
            control_dependencies = []
            target_labels = tf.stop_gradient(-tf_target_get_action_values)
            control_dependencies += [tf.assert_greater_equal(target_labels, 0., name='cost_assert_0')]
            control_dependencies += [tf.assert_less_equal(target_labels, 1., name='cost_assert_1')]
            with tf.control_dependencies(control_dependencies):
                tf_labels = tf.cast(tf_dones_ph, tf.float32) * tf_labels + \
                            (1 - tf.cast(tf_dones_ph, tf.float32)) * tf.maximum(tf_labels, target_labels)
           
            if loss == 'mse':
                cost = tf.reduce_sum(tf.square(tf_train_values - tf_labels), axis=2, keep_dims=True) 
            else:
                raise NotImplementedError
            return cost, control_dependencies

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = MACPolicy.create_session_and_graph(gpu_device=self._gpu_device, gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            if ext.get_seed() is not None:
                ext.set_seed(ext.get_seed())

            ### create input output placeholders
            tf_obs_im_ph, tf_obs_vec_ph, tf_actions_ph, tf_dones_ph, tf_rewards_ph, \
                tf_obs_im_target_ph, tf_obs_vec_target_ph, tf_test_es_ph_dict, \
                tf_episode_timesteps_ph = self._graph_input_output_placeholders()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            ### policy
            policy_scope = 'policy'
            with tf.variable_scope(policy_scope):
                ### create preprocess placeholders
                tf_preprocess = self._graph_preprocess_placeholders()

                tf_obs_vec, tf_obs_vec_target, tf_final_add, tf_final_add_target, tf_goal_vec = self._graph_extract_from_obs_vec_ph(tf_obs_vec_ph, tf_obs_vec_target_ph)

                ### process obs to lowd
                tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_im_ph, tf_obs_vec, tf_preprocess, is_training=True)
                ### create training policy
                tf_train_values, tf_train_values_softmax, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_final_add, tf_actions_ph[:, :self._H, :],
                                          self._values_softmax, tf_preprocess, is_training=True)

            with tf.variable_scope(policy_scope, reuse=True):
                tf_train_values_test, tf_train_values_softmax_test, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_final_add, tf_actions_ph[:, :self._get_action_test['H'], :],
                                          self._values_softmax, tf_preprocess, is_training=False)
                # tf_get_value = tf.reduce_sum(tf_train_values_softmax_test * tf_train_values_test, reduction_indices=1)
#                tf_get_value = -tf.sigmoid(tf_train_values_test) if self._is_classification else -tf_train_values_test
                # TODO
                tf_get_value = -tf_train_values_test

            ### action selection
            tf_get_action, tf_get_action_value, tf_get_action_reset_ops = \
                self._graph_get_action(tf_obs_im_ph, tf_obs_vec, tf_final_add, tf_goal_vec,
                                       self._get_action_test, policy_scope, True, policy_scope, True,
                                       tf_episode_timesteps_ph=tf_episode_timesteps_ph)
            ### exploration strategy and logprob
            tf_get_action_explore = self._graph_get_action_explore(tf_get_action, tf_test_es_ph_dict)

            ### get policy variables
            tf_policy_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      scope=policy_scope), key=lambda v: v.name)
            tf_trainable_policy_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                scope=policy_scope), key=lambda v: v.name)

            # TODO come back to target stuff later
            ### create target network
            if self._use_target:
                target_scope = 'target' if self._separate_target_params else 'policy'
                ### action selection
                tf_obs_im_target_ph_packed = tf.concat([tf_obs_im_target_ph[:, h - self._obs_history_len:h, :]
                                                        for h in range(self._obs_history_len, self._obs_history_len + self._N + 1)],
                                                       0)
                tf_target_get_action, tf_target_get_action_values, _ = self._graph_get_action(tf_obs_im_target_ph_packed,
                                                                                              tf_obs_vec_target_ph,
                                                                                              self._get_action_target,
                                                                                              scope_select=policy_scope,
                                                                                              reuse_select=True,
                                                                                              scope_eval=target_scope,
                                                                                              reuse_eval=(target_scope == policy_scope),
                                                                                              tf_episode_timesteps_ph=None,) # TODO: would need to fill in

                tf_target_get_action_values = tf.transpose(tf.reshape(tf_target_get_action_values, (self._N + 1, -1)))[:, 1:]
            else:
                tf_target_get_action_values = tf.zeros([tf.shape(tf_train_values)[0], self._N])

            ### update target network
            if self._use_target and self._separate_target_params:
                tf_policy_vars_nobatchnorm = list(
                    filter(lambda v: 'biased' not in v.name and 'local_step' not in v.name,
                           tf_policy_vars))
                tf_target_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope=target_scope), key=lambda v: v.name)
                assert (len(tf_policy_vars_nobatchnorm) == len(tf_target_vars))
                tf_update_target_fn = []
                for var, var_target in zip(tf_policy_vars_nobatchnorm, tf_target_vars):
                    assert (var.name.replace(policy_scope, '') == var_target.name.replace(target_scope, ''))
                    tf_update_target_fn.append(var_target.assign(var))
                tf_update_target_fn = tf.group(*tf_update_target_fn)
            else:
                tf_target_vars = None
                tf_update_target_fn = None

            ### optimization
#            tf_cost, tf_mse = self._graph_cost(tf_train_values, tf_train_values_softmax, tf_rewards_ph, tf_dones_ph,
#                                               tf_obs_im_target_ph, tf_obs_vec_target_ph, tf_target_get_action_values)
            tf_cost, tf_mse, tf_costs = self._graph_cost(tf_train_values, tf_rewards_ph, tf_dones_ph, tf_obs_im_target_ph,
                                               tf_obs_vec_target_ph, tf_target_get_action_values)
            tf_opt, tf_lr_ph = self._graph_optimize(tf_cost, tf_trainable_policy_vars)

            ### initialize
            self._graph_init_vars(tf_sess)
            
            ### savers
            tf_saver_inference = tf.train.Saver(tf_policy_vars, max_to_keep=None)
            tf_saver_train = tf.train.Saver(max_to_keep=None) if not self._inference_only else None
            
        # TODO
        ### what to return
        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'obs_im_ph': tf_obs_im_ph,
            'obs_vec_ph': tf_obs_vec_ph,
            'actions_ph': tf_actions_ph,
            'dones_ph': tf_dones_ph,
            'rewards_ph': tf_rewards_ph,
            'obs_im_target_ph': tf_obs_im_target_ph,
            'obs_vec_target_ph': tf_obs_vec_target_ph,
            'test_es_ph_dict': tf_test_es_ph_dict,
            'episode_timesteps_ph': tf_episode_timesteps_ph,
            'preprocess': tf_preprocess,
            'get_value': tf_get_value,
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
            'saver_inference': tf_saver_inference,
            'saver_train': tf_saver_train
        }

    ################
    ### Training ###
    ################

    def train_step(self, step, steps, observations, actions, rewards, values, dones, logprobs, use_target):
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
            self._tf_dict['dones_ph']: np.logical_or(not use_target, dones[:, :self._N]),
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
        }
        feed_dict[self._tf_dict['obs_vec_target_ph']] = observations_vec
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
        for i, sensor in enumerate(self._output_sensors):
            self._log_stats['{0} cost'.format(sensor['name'])].append(costs[i])
