import os, itertools
from collections import defaultdict
from collections import OrderedDict
import numpy as np

from gcg.envs.spaces.discrete import Discrete
from gcg.policies.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from gcg.policies.exploration_strategies.gaussian_strategy import GaussianStrategy
from gcg.policies.tf import networks
from gcg.policies.tf import tf_utils
from gcg.misc import schedules
from gcg.data.logger import logger

import rospy

from sensor_msgs.msg import Joy
from crazyflie.msg import CFCommand
from crazyflie.msg import CFMotion


THROTTLE_AXIS = 5 # up 1
ROLL_AXIS = 2 #left 1
PITCH_AXIS = 3 #up 1
YAW_AXIS = 0 #left 1

#RP motion
THROTTLE_SCALE = 0.1
ROLL_SCALE = 0.5
PITCH_SCALE = 0.5
YAW_SCALE = -120

#standard motion
VX_SCALE = 0.5
VY_SCALE = 0.5

TAKEOFF_CHANNEL = 7 #RT
ESTOP_CHANNEL = 2 #B
LAND_CHANNEL = 6 #LT
UNLOCK_ESTOP_CHANNEL = 0 #X

TOLERANCE = 0.05
ALT_TOLERANCE = 0.08


class CrazyflieTeleopPolicy(object):
    def __init__(self, **kwargs):

        #used to get joystick input
        rospy.init_node("CrazyflieTeleopPolicy", anonymous=True)

        self._outputs = kwargs['outputs'] 
        self._joy_topic = kwargs['joy_topic']

        self.curr_joy = None
        self.cmd = -1 # -1 : NONE

        self.is_flow_motion = flow_motion

        rospy.Subscriber(self._joy_topic, Joy, self.joy_cb)
        # self._rew_fn = kwargs['rew_fn']
        
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
        
        # ### model horizons
        # self._N = kwargs['N'] # number of returns to use (N-step)
        # self._H = kwargs['H'] # action planning horizon for training
        # self._gamma = kwargs['gamma'] # reward decay
        # self._obs_history_len = kwargs['obs_history_len'] # how many previous observations to use

        # ### model architecture
        # self._inference_only = kwargs.get('inference_only', False)
        # self._image_graph = kwargs['image_graph']
        # self._observation_graph = kwargs['observation_graph']
        # self._action_graph = kwargs['action_graph']
        # self._rnn_graph = kwargs['rnn_graph']
        # self._output_graph = kwargs['output_graph']
        # ### scopes
        # self._image_scope = 'image_scope'
        # self._observation_scope = 'observation_scope'
        # self._action_scope = 'action_scope'
        # self._rnn_scope = 'rnn_scope'
        # self._output_scope = 'output_scope'

        # ### target network
        # self._use_target = kwargs['use_target']
        # self._separate_target_params = kwargs['separate_target_params']
        # ### training
        # self._only_completed_episodes = kwargs['only_completed_episodes']
        # self._weight_decay = kwargs['weight_decay']
        # self._lr_schedule = schedules.PiecewiseSchedule(**kwargs['lr_schedule'])
        # self._grad_clip_norm = kwargs['grad_clip_norm']
        # self._gpu_device = kwargs['gpu_device']
        # self._gpu_frac = kwargs['gpu_frac']

        # ### action selection and exploration
        # self._get_action_test = kwargs['get_action_test']
        # self._get_action_target = kwargs['get_action_target']
        # assert(self._get_action_target['type'] == 'random')
        # gaussian_es_params = kwargs['exploration_strategies'].get('GaussianStrategy', None)
        # if gaussian_es_params is not None:
        #     self._gaussian_es = GaussianStrategy(self._env_spec, **gaussian_es_params) if gaussian_es_params else None
        # else:
        #     self._gaussian_es = None
        # epsilon_greedy_es_params = kwargs['exploration_strategies'].get('EpsilonGreedyStrategy', None)
        # if epsilon_greedy_es_params is not None:
        #     self._epsilon_greedy_es = EpsilonGreedyStrategy(self._env_spec, **epsilon_greedy_es_params)
        # else:
        #     self._epsilon_greedy_es = None

        # ### setup the model
        # self._tf_debug = dict()
        # self._tf_dict = self._graph_setup()

        # ### logging
        # self._log_stats = defaultdict(list)

        # assert(self._N >= self._H)

    #################
    ### Callbacks ###
    #################
    def dead_band(self, signal):
        new_axes = [0] * len(signal.axes)
        for i in range(len(signal.axes)):
            new_axes[i] = signal.axes[i] if abs(signal.axes[i]) > TOLERANCE else 0
        signal.axes = new_axes

    def joy_cb(msg):
        if self.curr_joy:
            if msg.buttons[ESTOP_CHANNEL] and not self.curr_joy.buttons[ESTOP_CHANNEL]:
                #takeoff
                self.cmd = CFCommand.ESTOP
                print("CALLING ESTOP")
            elif msg.buttons[TAKEOFF_CHANNEL] and not self.curr_joy.buttons[TAKEOFF_CHANNEL]:
                #takeoff
                self.cmd = CFCommand.TAKEOFF
                print("CALLING TAKEOFF")
            elif msg.buttons[LAND_CHANNEL] and not self.curr_joy.buttons[LAND_CHANNEL]:
                #takeoff
                self.cmd = CFCommand.LAND
                print("CALLING LAND")
        else:
            if msg.buttons[ESTOP_CHANNEL] :
                #takeoff
                self.cmd = CFCommand.ESTOP
                print("CALLING ESTOP")
            elif msg.buttons[TAKEOFF_CHANNEL] :
                #takeoff
                self.cmd = CFCommand.TAKEOFF
                print("CALLING TAKEOFF")
            elif msg.buttons[LAND_CHANNEL] :
                #takeoff
                self.cmd = CFCommand.LAND
                print("CALLING LAND")
            '''

        if self.curr_joy:
            if msg.buttons[ESTOP_CHANNEL] and not self.curr_joy.buttons[ESTOP_CHANNEL]:
                #takeoff
                self.cmd = CFCommand.ESTOP
            elif msg.axes[2] and not self.curr_joy.axes[2]:
                #takeoff
                self.cmd = CFCommand.TAKEOFF
            elif msg.axes[5] and not self.curr_joy.axes[5]:
                #takeoff
                self.cmd = CFCommand.LAND'''
        self.dead_band(msg)
        self.curr_joy = msg

    ##################
    ### Properties ###
    ##################

    @property
    def obs_history_len(self):
        return self._obs_history_len

    ################
    ### Training ###
    ################

    def update_target(self):
        return

    def train_step(self, step, steps, observations, goals, actions, rewards, dones, use_target):
        """
        :param steps: [batch_size, N+1]
        :param observations_im: [batch_size, N+1 + obs_history_len-1, obs_im_dim]
        :param observations_vec: [batch_size, N+1 + obs_history_len-1, obs_vec_dim]
        :param actions: [batch_size, N+1, action_dim]
        :param rewards: [batch_size, N+1]
        :param dones: [batch_size, N+1]
        """
        return

    def reset_weights(self):
        tf_sess = self._tf_dict['sess']
        tf_graph = tf_sess.graph
        with tf_sess.as_default(), tf_graph.as_default():
            self._graph_init_vars(tf_sess)

    ######################
    ### Policy methods ###
    ######################

    def get_action(self, step, current_episode_step, observation, goal, explore):
        # chosen_actions, chosen_values, action_infos = self.get_actions([step], [current_episode_step] [observation],
        #                                                               [goal], explore=explore)
        # return chosen_actions[0], chosen_values[0], action_infos[0]

        if self.cmd != -1:
            motion = CFCommand()
            if self.cmd == CFCommand.ESTOP:
                motion.cmd = CFCommand.ESTOP

            elif self.cmd == CFCommand.TAKEOFF:
                motion.cmd = CFCommand.TAKEOFF

            elif self.cmd == CFCommand.LAND:
                motion.cmd = CFCommand.LAND

            #reset
            self.cmd = -1

        #repeat send at 10Hz
        elif self.curr_joy:
            motion = CFMotion()

            motion.is_flow_motion = self.is_flow_motion
                # computing regular vx, vy, yaw, alt motion

            if self.is_flow_motion:
                motion.y = self.curr_joy.axes[ROLL_AXIS] * VY_SCALE
                motion.x = self.curr_joy.axes[PITCH_AXIS] * VX_SCALE
            else:
                motion.y = self.curr_joy.axes[ROLL_AXIS] * ROLL_SCALE
                motion.x = self.curr_joy.axes[PITCH_AXIS] * PITCH_SCALE

            #common
            motion.yaw = self.curr_joy.axes[YAW_AXIS] * YAW_SCALE

                
            # print(self.curr_joy.axes)
            motion.dz = self.curr_joy.axes[THROTTLE_AXIS]* THROTTLE_SCALE
            # print("ALT CHANGE: %.3f" % motion.dz)

                #what is self.alt and where is it used??
                #self.alt = motion.alt_change

                # motion.vx = self.curr_joy.axes[3] * 0.1
                # motion.vy = self.curr_joy.axes[4] * 0.1
                # motion.yaw = self.curr_joy.axes[6] * 0.1
            
        # return motion


    
    def reset_get_action(self):
        return

    def terminate(self):
        return

    #####################
    ### Model methods ###
    #####################

    def get_model_outputs(self, observations, actions):
        return

    ######################
    ### Saving/loading ###
    ######################

    # def _saver_ckpt_name(self, ckpt_name, saver_name):
    #     name, ext = os.path.splitext(ckpt_name)
    #     saver_ckpt_name = '{0}_{1}{2}'.format(name, saver_name, ext)
    #     return saver_ckpt_name

    # def save(self, ckpt_name, train=True):
    #     if train:
    #         savers_keys = [k for k in self._tf_dict['savers_dict'].keys() if 'inference' not in k]
    #     else:
    #         savers_keys = ['inference']

    #     for saver_name in savers_keys:
    #         saver = self._tf_dict['savers_dict'][saver_name]
    #         saver.save(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, saver_name), write_meta_graph=False)

    # def restore(self, ckpt_name, train=True, train_restore=('train',)):
    #     """
    #     :param: train_restore: 'train', 'image', 'observation', 'action', 'rnn', 'output
    #     """
    #     savers_keys = train_restore if train else ['inference']

    #     for saver_name in savers_keys:
    #         saver = self._tf_dict['savers_dict'][saver_name]
    #         saver.restore(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, saver_name))

    ###############
    ### Logging ###
    ###############

    # def log(self):
    #     for k in sorted(self._log_stats.keys()):
    #         if k == 'Depth':
    #             logger.record_tabular(k+'Mean', np.mean(self._log_stats[k]))
    #             logger.record_tabular(k+'Std', np.std(self._log_stats[k]))
    #         else:
    #             logger.record_tabular(k, np.mean(self._log_stats[k]))
    #     self._log_stats.clear()
