import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

import rllab.misc.logger as rllab_logger

from sandbox.gkahn.tf.core import xplatform

from sandbox.gkahn.traversability.tf import fcn

class TraversabilityPolicy:

    def __init__(self, **kwargs):
        ### environment
        self._env_spec = kwargs['env_spec']

        ### training
        self._num_classes = 2
        self._inference_only = kwargs.get('inference_only', False)
        self._lr = kwargs['lr']
        self._weight_decay = kwargs['weight_decay']
        self._gpu_device = kwargs['gpu_device']
        self._gpu_frac = kwargs['gpu_frac']

        ### setup the model
        self._tf_debug = dict()
        self._tf_dict = self._graph_setup()

        ### action selection
        self._internal_get_steer = None

        ### logging
        self._log_stats = defaultdict(list)

    ##################
    ### Properties ###
    ##################

    @property
    def N(self):
        return 1

    @property
    def gamma(self):
        return 1

    @property
    def session(self):
        return self._tf_dict['sess']

    @property
    def _obs_is_im(self):
        return len(self._env_spec.observation_space.shape) > 1

    @property
    def obs_history_len(self):
        return 1

    @property
    def only_completed_episodes(self):
        return False

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
        assert(self._obs_is_im)
        obs_shape = list(self._env_spec.observation_space.shape)
        obs_dtype = tf.uint8

        with tf.variable_scope('input_output_placeholders'):
            tf_obs_ph = tf.placeholder(obs_dtype, [None] + obs_shape, name='tf_obs_ph')
            tf_labels_ph = tf.placeholder(obs_dtype, [None] + obs_shape[:-1], name='tf_labels_ph')

        return tf_obs_ph, tf_labels_ph

    def _graph_inference(self, tf_obs_ph):
        ### create network
        network = fcn.FCN16VGG()
        rgb = tf.cast(tf.tile(tf_obs_ph, (1, 1, 1, 3)), tf.float32)
        network.build(rgb, train=not self._inference_only, num_classes=self._num_classes)

        ### get relevant outputs
        tf_scores = network.upscore32
        tf_probs = network.prob32

        return tf_scores, tf_probs

    def _graph_cost(self, tf_labels_ph, tf_scores):
        num_classes = self._num_classes
        head = None

        logits = tf.reshape(tf_scores, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.to_float(tf.reshape(tf.one_hot(tf_labels_ph, num_classes, axis=3), (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1]) # TODO: change to built in softmax with logits

        tf_mse = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            tf_weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            tf_weight_decay = 0
        tf_cost = tf_mse + tf_weight_decay

        return tf_cost, tf_mse

    def _graph_optimize(self, tf_cost, tf_policy_vars):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-4)
            gradients = optimizer.compute_gradients(tf_cost, var_list=tf_policy_vars)
            tf_opt = optimizer.apply_gradients(gradients)
        return tf_opt

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = TraversabilityPolicy.create_session_and_graph(gpu_device=self._gpu_device,
                                                                              gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            ### create input output placeholders
            tf_obs_ph, tf_labels_ph = self._graph_input_output_placeholders()

            ### inference
            policy_scope = 'policy'
            with tf.variable_scope(policy_scope):
                tf_scores, tf_probs = self._graph_inference(tf_obs_ph)

            ### get policy variables
            tf_policy_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      scope=policy_scope), key=lambda v: v.name)
            tf_trainable_policy_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                scope=policy_scope), key=lambda v: v.name)

            if not self._inference_only:
                ### cost and optimize
                tf_cost, tf_mse = self._graph_cost(tf_labels_ph, tf_scores)
                tf_opt = self._graph_optimize(tf_cost, tf_trainable_policy_vars)
            else:
                tf_cost = tf_mse = tf_opt = None

            ### savers
            tf_saver_inference = tf.train.Saver(tf_policy_vars, max_to_keep=None)
            tf_saver_train = tf.train.Saver(max_to_keep=None) if not self._inference_only else None

            ### initialize
            tf_sess.run([tf.global_variables_initializer()])

        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'obs_ph': tf_obs_ph,
            'labels_ph': tf_labels_ph,
            'scores': tf_scores,
            'probs': tf_probs,
            'cost': tf_cost,
            'mse': tf_mse,
            'opt': tf_opt,
            'saver_inference': tf_saver_inference,
            'saver_train': tf_saver_train,
            'policy_vars': tf_policy_vars
        }

    ################
    ### Training ###
    ################

    def update_preprocess(self, preprocess_stats):
        pass

    def update_target(self):
        pass

    def train_step(self, step, steps, observations, actions, rewards, values, dones, logprobs, use_target):
        pass # don't update the model during RL

    def reset_weights(self):
        tf_sess = self._tf_dict['sess']
        tf_graph = tf_sess.graph
        with tf_sess.as_default(), tf_graph.as_default():
            self._graph_init_vars(tf_sess)

    ######################
    ### Policy methods ###
    ######################

    def get_action(self, step, current_episode_step, observation, explore):
        chosen_actions, chosen_values, action_info = self.get_actions([step], [current_episode_step] [observation],
                                                                      explore=explore)
        return chosen_actions[0], chosen_values[0], action_info

    def _default_get_steer(self, prob_coll):
        obs_shape = list(self._env_spec.observation_space.shape)

        ind_horiz, ind_vert = np.meshgrid(np.arange(obs_shape[1]), np.arange(obs_shape[0]))

        weights_vert = ind_vert/ ind_vert.sum(axis=0, keepdims=True)

        weights_horiz = np.roll(ind_horiz, obs_shape[1] // 2, axis=1)
        weights_horiz = abs(weights_horiz - weights_horiz.mean())
        weights_horiz = weights_horiz / weights_horiz.sum(axis=1, keepdims=True)

        weights = weights_vert + 0.5 * (obs_shape[0] / float(obs_shape[1])) * weights_horiz
        weights /= weights.sum()

        cost_map = prob_coll * weights
        kp = 0.15
        steer = kp * (cost_map.sum(axis=0).argmin(axis=0) - obs_shape[1]//2) / float(obs_shape[1]//2)

        return steer

    def _get_steer(self, prob_coll):
        if self._internal_get_steer is None:
            return self._default_get_steer(prob_coll)
        else:
            return self._internal_get_steer(prob_coll)

    def set_get_steer(self, get_steer_func):
        self._internal_get_steer = get_steer_func

    def get_actions(self, steps, current_episode_steps, observations, explore):
        obs_shape = list(self._env_spec.observation_space.shape)
        observations = np.reshape(observations, [len(steps)] + obs_shape)

        feed_dict = {
            self._tf_dict['obs_ph']: observations
        }

        probs, = self._tf_dict['sess'].run([self._tf_dict['probs']], feed_dict=feed_dict)
        probs_coll = probs[:, :, :, 1]

        steers = [self._get_steer(prob_coll) for prob_coll in probs_coll]
        speeds = np.ones(len(steps), dtype=float) * self._env_spec.action_space.high[1]
        actions = np.vstack((steers, speeds)).T
        actions = np.clip(actions, *self._env_spec.action_space.bounds)


        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(weights, cmap='inferno')
        #
        # plt.figure()
        # plt.imshow(1 - probs_coll[0], cmap='Greys_r', vmin=0, vmax=1)
        # plt.show()
        #
        # import IPython; IPython.embed()

        # actions = [self._env_spec.action_space.sample() for _ in steps]
        values = [np.nan] * len(steps)
        logprobs = [np.nan] * len(steps)
        action_infos = [{'prob_coll': prob_coll} for prob_coll in probs_coll]# [{}] * len(steps)

        return actions, values, logprobs, action_infos

    def reset_get_action(self):
        pass

    @property
    def recurrent(self):
        return False

    def terminate(self):
        self._tf_dict['sess'].close()

    #####################
    ### Model methods ###
    #####################

    def get_model_outputs(self, observations, actions):
        feed_dict = {
            self._tf_dict['obs_ph']: observations
        }

        probs,  = self._tf_dict['sess'].run([self._tf_dict['probs']], feed_dict=feed_dict)

        return probs

    ######################
    ### Saving/loading ###
    ######################

    def get_params_internal(self, **tags):
        with self._tf_dict['graph'].as_default():
            return sorted(tf.get_collection(xplatform.global_variables_collection_name()), key=lambda v: v.name)

    def save(self, ckpt_name, train=True):
        saver = self._tf_dict['saver_train'] if train else self._tf_dict['saver_inference']
        saver.save(self._tf_dict['sess'], ckpt_name, write_meta_graph=False)

    def restore(self, ckpt_name, train=True):
        saver = self._tf_dict['saver_train'] if train else self._tf_dict['saver_inference']
        saver.restore(self._tf_dict['sess'], ckpt_name)

    ###############
    ### Logging ###
    ###############

    def log(self):
        for k in sorted(self._log_stats.keys()):
            if k == 'Depth':
                rllab_logger.record_tabular(k+'Mean', np.mean(self._log_stats[k]))
                rllab_logger.record_tabular(k+'Std', np.std(self._log_stats[k]))
            else:
                rllab_logger.record_tabular(k, np.mean(self._log_stats[k]))
        self._log_stats.clear()

    ########################
    ### Offline training ###
    ########################

    def train_step_offline(self, images, labels):
        feed_dict = {
            self._tf_dict['obs_ph']: images,
            self._tf_dict['labels_ph']: labels
        }

        cost, mse, _ = self._tf_dict['sess'].run([self._tf_dict['cost'],
                                                  self._tf_dict['mse'],
                                                  self._tf_dict['opt']],
                                                 feed_dict=feed_dict)
        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)
