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
        self._inference_only = kwargs.get('inference_only', False)
        self._weight_decay = kwargs['weight_decay']
        self._gpu_device = kwargs['gpu_device']
        self._gpu_frac = kwargs['gpu_frac']

        ### setup the model
        self._tf_debug = dict()
        self._tf_dict = self._graph_setup()

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
        obs_shape = self._env_spec.observation_space.shape
        obs_dtype = tf.uint8

        with tf.variable_scope('input_output_placeholders'):
            tf_obs_ph = tf.placeholder(obs_dtype, [None] + obs_shape, name='tf_obs_ph')
            tf_labels_ph = tf.placeholder(obs_dtype, [None] + obs_shape, name='tf_labels_ph')

        return tf_obs_ph, tf_labels_ph

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

    def get_actions(self, steps, current_episode_steps, observations, explore):
        pass # TODO

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
        pass # TODO

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
