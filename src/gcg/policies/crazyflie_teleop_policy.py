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
        # rospy.init_node("CrazyflieTeleopPolicy", anonymous=True)

        # self._outputs = kwargs['outputs'] 
        self._joy_topic = kwargs['joy_topic']

        self.curr_joy = None
        self.cmd = -1 # -1 : NONE

        self.is_flow_motion = bool(kwargs['flow_motion'])

        rospy.Subscriber(self._joy_topic, Joy, self.joy_cb)
        # self._rew_fn = kwargs['rew_fn']

        self._N = kwargs['N']
        self._gamma = kwargs['gamma']
        
        ### environment
        self._env_spec = kwargs['env_spec']
        self._obs_vec_keys = list(self._env_spec.observation_vec_spec.keys())
        self._action_keys = list(self._env_spec.action_spec.keys())
        self._goal_keys = list(self._env_spec.goal_spec.keys())
        # self._output_keys = sorted([output['name'] for output in self._outputs])
        self._obs_im_shape = self._env_spec.observation_im_space.shape

        
        self._obs_im_dim = np.prod(self._obs_im_shape)
        self._obs_vec_dim = len(self._obs_vec_keys)
        self._action_dim = len(self._action_keys)
        self._goal_dim = len(self._goal_keys)
        # self._output_dim = len(self._output_keys)
        

    #################
    ### Callbacks ###
    #################
    def dead_band(self, signal):
        new_axes = [0] * len(signal.axes)
        for i in range(len(signal.axes)):
            new_axes[i] = signal.axes[i] if abs(signal.axes[i]) > TOLERANCE else 0
        signal.axes = new_axes

    def joy_cb(self, msg):
        # if self.curr_joy:
        #     if msg.buttons[ESTOP_CHANNEL] and not self.curr_joy.buttons[ESTOP_CHANNEL]:
        #         #takeoff
        #         self.cmd = CFCommand.ESTOP
        #         print("CALLING ESTOP")
        #     elif msg.buttons[TAKEOFF_CHANNEL] and not self.curr_joy.buttons[TAKEOFF_CHANNEL]:
        #         #takeoff
        #         self.cmd = CFCommand.TAKEOFF
        #         print("CALLING TAKEOFF")
        #     elif msg.buttons[LAND_CHANNEL] and not self.curr_joy.buttons[LAND_CHANNEL]:
        #         #takeoff
        #         self.cmd = CFCommand.LAND
        #         print("CALLING LAND")
        # else:
        #     if msg.buttons[ESTOP_CHANNEL] :
        #         #takeoff
        #         self.cmd = CFCommand.ESTOP
        #         print("CALLING ESTOP")
        #     elif msg.buttons[TAKEOFF_CHANNEL] :
        #         #takeoff
        #         self.cmd = CFCommand.TAKEOFF
        #         print("CALLING TAKEOFF")
        #     elif msg.buttons[LAND_CHANNEL] :
        #         #takeoff
        #         self.cmd = CFCommand.LAND
        #         print("CALLING LAND")
        self.dead_band(msg)
        self.curr_joy = msg

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
    def obs_history_len(self):
        return 1
    ################
    ### Training ###
    ################

    ######################
    ### Policy methods ###
    ######################

    def get_action(self, step, current_episode_step, observation, goal, explore):
        # chosen_actions, chosen_values, action_infos = self.get_actions([step], [current_episode_step] [observation],
        #                                                               [goal], explore=explore)
        # return chosen_actions[0], chosen_values[0], action_infos[0]

        motion = CFMotion()

        # if self.cmd != -1:
        #     motion = CFCommand()
        #     if self.cmd == CFCommand.ESTOP:
        #         motion.cmd = CFCommand.ESTOP

        #     elif self.cmd == CFCommand.TAKEOFF:
        #         motion.cmd = CFCommand.TAKEOFF

        #     elif self.cmd == CFCommand.LAND:
        #         motion.cmd = CFCommand.LAND

        #     #reset
        #     self.cmd = -1

        #repeat send at 10Hz
        # print(self.curr_joy)
        if self.curr_joy:

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
        return [ motion.x, motion.y, motion.yaw, motion.dz ]

    def get_actions(self, steps, current_episode_steps, observations, goals, explore):
        ds = [{} for _ in steps]
        observations_im, observations_vec = observations
        
        values = [0 for _ in steps]
        joy_action = self.get_action(None, None, None, None, None) #args don't matter
        actions = [joy_action] * len(steps)

        return actions, values, ds
    
    def reset_get_action(self):
        return

    def terminate(self):
        self.curr_joy = None
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

    def save(self, ckpt_name, train=True):
        # if train:
        #     savers_keys = [k for k in self._tf_dict['savers_dict'].keys() if 'inference' not in k]
        # else:
        #     savers_keys = ['inference']

        # for saver_name in savers_keys:
        #     saver = self._tf_dict['savers_dict'][saver_name]
        #     saver.save(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, saver_name), write_meta_graph=False)
        pass

    def restore(self, ckpt_name, train=True, train_restore=('train',)):
        """
        :param: train_restore: 'train', 'image', 'observation', 'action', 'rnn', 'output
        # """
        # savers_keys = train_restore if train else ['inference']

        # for saver_name in savers_keys:
        #     saver = self._tf_dict['savers_dict'][saver_name]
        #     saver.restore(self._tf_dict['sess'], self._saver_ckpt_name(ckpt_name, saver_name))
        pass

    ###############
    ### Logging ###
    ###############

    def log(self):
        # for k in sorted(self._log_stats.keys()):
        #     if k == 'Depth':
        #         logger.record_tabular(k+'Mean', np.mean(self._log_stats[k]))
        #         logger.record_tabular(k+'Std', np.std(self._log_stats[k]))
        #     else:
        #         logger.record_tabular(k, np.mean(self._log_stats[k]))
        # self._log_stats.clear()
        pass
