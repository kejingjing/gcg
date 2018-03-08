import os
from collections import OrderedDict
import numpy as np
# import cv2
from PIL import Image
from io import BytesIO

from gcg.envs.env_spec import EnvSpec
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.data.logger import logger

import time

try:
    import rospy
    import rosbag
    import std_msgs.msg
    import geometry_msgs.msg
    import sensor_msgs.msg
    import crazyflie.msg
    #from crazyflie.msg import CFMotion
    #from crazyflie.msg import CFCommand
    ROS_IMPORTED = True
except:
    ROS_IMPORTED = False

#print(ROS_IMPORTED)
print("Running Crazyflie Environment")

class RolloutRosbag:
    def __init__(self):
        self._rosbag = None
        self._last_write = None

    @property
    def _rosbag_dir(self):
        dir = os.path.join(logger.dir, 'rosbags')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    def _rosbag_name(self, num):
        return os.path.join(self._rosbag_dir, 'rosbag{0:04d}.bag'.format(num))

    @property
    def is_open(self):
        return (self._rosbag is not None)

    def open(self):
        assert (not self.is_open)

        bag_num = 0
        while os.path.exists(self._rosbag_name(bag_num)):
            bag_num += 1

        self._rosbag = rosbag.Bag(self._rosbag_name(bag_num), 'w')
        self._last_write = rospy.Time.now()

    def write(self, topic, msg, stamp):
        assert (self._rosbag is not None)

        if msg is not None and stamp is not None:
            if stamp > self._last_write:
                self._rosbag.write(topic, msg)
        else:
            logger.warn('Topic {0} not received'.format(topic))

    def write_all(self, topics, msg_dict, stamp_dict):
        for topic in topics:
            self.write(topic, msg_dict.get(topic), stamp_dict.get(topic))

    def close(self):
        assert (self.is_open)

        self._rosbag.close()
        self._rosbag = None
        self._last_write = None

    def trash(self):
        assert (self.is_open)

        bag_fname = self._rosbag.filename

        try:
            self.close()
        except:
            pass

        os.remove(os.path.join(self._rosbag_dir, bag_fname))


class CrazyflieEnv:
    def __init__(self, params={}):
        params.setdefault('dt', 0.25)
        params.setdefault('horizon', int(5. * 60. / params['dt']))  # 5 minutes worth
        params.setdefault('ros_namespace', '/crazyflie/')
        params.setdefault('obs_shape', (96, 72, 1))
        # params.setdefault('steer_limits', [-0.9, 0.9])
        params.setdefault('dz_limits', [0,0]) #default change in alt
        params.setdefault('yaw_limits', [0,0]) #default yaw rate range
        params.setdefault('velocity_limits', [-0.3, 0.3])
        # params.setdefault('backup_motor', -0.22)
        # params.setdefault('backup_duration', 1.6)
        # params.setdefault('backup_steer_range', (-0.8, 0.8))
        params.setdefault('press_enter_on_reset', False)

        params.setdefault('use_joy_commands', True)
        params.setdefault('joy_start_btn', 1) #A
        params.setdefault('joy_stop_btn', 2) #B
        params.setdefault('joy_pause_run_btn', 3) # Y
        params.setdefault('joy_topic', '/joy')

        self._use_vel = True
        self._obs_shape = params['obs_shape']
        # self._steer_limits = params['steer_limits']
        # self._speed_limits = params['speed_limits']
        self._dz_limits = params['dz_limits']
        self._yaw_limits = params['yaw_limits']
        self._velocity_limits = params['velocity_limits']

        #new
        self._fixed_alt = self._dz_limits[0] == self._dz_limits[1]

        self._collision_reward = params['collision_reward']
        self._collision_reward_only = params['collision_reward_only']

        self._dt = params['dt']
        self.horizon = params['horizon']

        # start stop and pause
        self._use_joy_commands = params['use_joy_commands']
        self._joy_topic = params['joy_topic']
        self._joy_stop_btn = params['joy_stop_btn']
        self._joy_start_btn = params['joy_start_btn']
        self._start_pressed = False
        self._stop_pressed = False
        self._joy_pause_run_btn = params['joy_pause_run_btn']
        self._pause = False
        self._curr_joy = None


        self._setup_spec()
        assert (self.observation_im_space.shape[-1] == 1 or self.observation_im_space.shape[-1] == 3)
        self.spec = EnvSpec(
            observation_im_space=self.observation_im_space,
            action_space=self.action_space,
            action_selection_space=self.action_selection_space,
            observation_vec_spec=self.observation_vec_spec,
            action_spec=self.action_spec,
            action_selection_spec=self.action_selection_spec,
            goal_spec=self.goal_spec)

        self._last_step_time = None
        self._is_collision = False
        # self._backup_motor = params['backup_motor']
        # self._backup_duration = params['backup_duration']
        # self._backup_steer_range = params['backup_steer_range']
        self._press_enter_on_reset = params['press_enter_on_reset']

        ### ROS
        if not ROS_IMPORTED:
            logger.warn('ROS not imported')
            return

        rospy.init_node('CrazyflieEnv', anonymous=True)
        rospy.sleep(1)

        self._ros_namespace = params['ros_namespace']
        self._ros_topics_and_types = dict([
            ('cf/0/image', sensor_msgs.msg.CompressedImage),
            ('cf/0/data', crazyflie.msg.CFData),
            ('cf/0/coll', std_msgs.msg.Bool),

            # ('mode', std_msgs.msg.Int32),
            # ('steer', std_msgs.msg.Float32),
            # ('motor', std_msgs.msg.Float32),
            # ('encoder/left', std_msgs.msg.Float32),
            # ('encoder/right', std_msgs.msg.Float32),
            # ('encoder/both', std_msgs.msg.Float32),
            # ('orientation/quat', geometry_msgs.msg.Quaternion),
            # ('orientation/rpy', geometry_msgs.msg.Vector3),
            # ('imu', geometry_msgs.msg.Accel),
            # ('collision/all', std_msgs.msg.Int32),
            # ('collision/flip', std_msgs.msg.Int32),
            # ('collision/jolt', std_msgs.msg.Int32),
            # ('collision/stuck', std_msgs.msg.Int32),
            # ('collision/bumper', std_msgs.msg.Int32),
            # ('cmd/steer', std_msgs.msg.Float32),
            # ('cmd/motor', std_msgs.msg.Float32),
            # ('cmd/vel', std_msgs.msg.Float32)
        ])
        self._ros_msgs = dict()
        self._ros_msg_times = dict()
        for topic, type in self._ros_topics_and_types.items():
            rospy.Subscriber(topic, type, self.ros_msg_update, (topic,))
        
        self._ros_motion_pub = rospy.Publisher("/cf/0/motion", crazyflie.msg.CFMotion, queue_size=10)
        self._ros_command_pub = rospy.Publisher("/cf/0/command", crazyflie.msg.CFCommand, queue_size=10)

        # separate from big list
        if self._use_joy_commands:
            logger.debug("Environment using joystick commands")
            self._ros_joy_sub = rospy.Subscriber(self._joy_topic, sensor_msgs.msg.Joy, self._joy_cb)

        # self._ros_vx_pub = rospy.Publisher(self._ros_namespace + 'cmd/vx', std_msgs.msg.Float32, queue_size=10)
        # self._ros_vy_pub = rospy.Publisher(self._ros_namespace + 'cmd/vy', std_msgs.msg.Float32, queue_size=10)
        # self._ros_yaw_pub = rospy.Publisher(self._ros_namespace + 'cmd/yaw', std_msgs.msg.Float32, queue_size=10)
        # self._ros_alt_pub = rospy.Publisher(self._ros_namespace + 'cmd/alt', std_msgs.msg.Float32, queue_size=10)
        # self._ros_motor_pub = rospy.Publisher(self._ros_namespace + 'cmd/motor', std_msgs.msg.Float32, queue_size=10)
        self._ros_pid_enable_pub = rospy.Publisher(self._ros_namespace + 'pid/enable', std_msgs.msg.Empty,
                                                   queue_size=10)
        self._ros_pid_disable_pub = rospy.Publisher(self._ros_namespace + 'pid/disable', std_msgs.msg.Empty,
                                                    queue_size=10)

        self._ros_rolloutbag = RolloutRosbag()
        self._t = 0
        
        
    def _setup_spec(self):
        self.action_spec = OrderedDict()
        self.action_selection_spec = OrderedDict()
        self.observation_vec_spec = OrderedDict()
        self.goal_spec = OrderedDict()

        self.action_spec['vx'] = Box(low=-1., high=1.)
        self.action_spec['vy'] = Box(low=-1, high=1)
        self.action_spec['yaw'] = Box(low=-180, high=180)
        self.action_spec['dz'] = Box(low=-0.8, high=0.8) #meters
        self.action_space = Box(low=np.array([self.action_spec['vx'].low[0], self.action_spec['vy'].low[0], self.action_spec['yaw'].low[0], self.action_spec['dz'].low[0]]),
                                high=np.array([self.action_spec['vx'].high[0], self.action_spec['vy'].high[0], self.action_spec['yaw'].high[0], self.action_spec['dz'].high[0]]))

        self.action_selection_spec['vx'] = Box(low=self._velocity_limits[0], high=self._velocity_limits[1])
        self.action_selection_spec['vy'] = Box(low=self._velocity_limits[0], high=self._velocity_limits[1])
        self.action_selection_spec['yaw'] = Box(low=self._yaw_limits[0], high=self._yaw_limits[1])
        self.action_selection_spec['dz'] = Box(low=self._dz_limits[0], high=self._dz_limits[1]) #fixed
        self.action_selection_space = Box(low=np.array([self.action_selection_spec['vx'].low[0], self.action_selection_spec['vy'].low[0], self.action_selection_spec['yaw'].low[0], self.action_selection_spec['dz'].low[0]]),
                                high=np.array([self.action_selection_spec['vx'].high[0], self.action_selection_spec['vy'].high[0], self.action_selection_spec['yaw'].high[0], self.action_selection_spec['dz'].high[0]]))


        #Box(low=np.array([self.action_selection_spec['steer'].low[0],
                                          #               self.action_selection_spec['speed'].low[0]]),
                                          # high=np.array([self.action_selection_spec['steer'].high[0],
                                          #                self.action_selection_spec['speed'].high[0]]))

        assert (np.logical_and(self.action_selection_space.low >= self.action_space.low,
                               self.action_selection_space.high <= self.action_space.high).all())

        self.observation_im_space = Box(low=0, high=255, shape=self._obs_shape)
        self.observation_vec_spec['coll'] = Discrete(1)
        self.observation_vec_spec['accel_x'] = Box(low=-3, high=3) #3 is a guess
        self.observation_vec_spec['accel_y'] = Box(low=-3, high=3)
        self.observation_vec_spec['accel_z'] = Box(low=-3., high=3)
        self.observation_vec_spec['v_batt'] = Box(low=0, high=6)
        self.observation_vec_spec['alt'] = Box(low=0, high=1.2) #meters



    def _get_observation(self):
        msg = self._ros_msgs['cf/0/image']

        recon_pil_jpg = BytesIO(msg.data)
        recon_pil_arr = Image.open(recon_pil_jpg)

        is_grayscale = (self.observation_im_space.shape[-1] == 1)
        if is_grayscale:
            grayscale = recon_pil_arr.convert('L')
            grayscale_resized = grayscale.resize(self.observation_im_space.shape[:-1][::-1],
                                                 Image.ANTIALIAS)  # b/c (width, height)
            im = np.expand_dims(np.array(grayscale_resized), 2)
        else:
            # rgb = np.array(recon_pil_arr)
            rgb_resized = recon_pil_arr.resize(self.observation_im_space.shape[:-1][::-1],
                                               Image.ANTIALIAS)  # b/c (width, height)
            im = np.array(rgb_resized)

        coll = self._is_collision

        data = self._ros_msgs['cf/0/data']
        # accel_x = data.accel_x
        # accel_y = data.accel_y
        # accel_z = data.accel_z
        # v_batt = data.v_batt
        # alt = data.alt

        vec = np.array([coll, data.alt, data.v_batt, data.accel_x, data.accel_y, data.accel_z])


        return im, vec

    def _get_goal(self):
        # TODO: make sure if there is a goal, to add it as a ROS msg
        return np.array([])

    '''def _get_speed(self):
        return #self._ros_msgs['encoder/both'].data'''

    def _get_reward(self):
        if self._is_collision:
            reward = self._collision_reward
        else:
            reward = 0
        return reward

    ### JOYSTICK ###

    def _get_joy_stop(self):
        #in this order
        return self._use_joy_commands and self._stop_pressed

    def _get_pause(self):
        return self._use_joy_commands and self._pause

    def _get_joy_start(self):
        #in this order
        return self._use_joy_commands and self._start_pressed
        self._curr_joy and self._curr_joy.buttons[self._joy_start_btn]

    ### OTHER ###
    
    def _get_collision(self):
        return self._is_collision 

    def _get_done(self):
        # if self._get_joy_stop():
        #     print("STOPPING MANUALLY")
        #     print(self._curr_joy.buttons)
        #     print(self._joy_stop_btn)
        return self._get_joy_stop() or self._get_collision()

    '''    def _set_steer(self, steer):
        self._ros_steer_pub.publish(std_msgs.msg.Float32(steer))

    def _set_vel(self, vel):
        if self._is_collision and vel > 0:
            vel = 0.
        self._ros_vel_pub.publish(std_msgs.msg.Float32(vel))

    def _set_motor(self, motor, duration):
        self._ros_pid_disable_pub.publish(std_msgs.msg.Empty())
        rospy.sleep(0.25)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < duration:
            self._ros_motor_pub.publish(std_msgs.msg.Float32(motor))
            rospy.sleep(0.01)
        self._ros_motor_pub.publish(std_msgs.msg.Float32(0.))
        self._ros_pid_enable_pub.publish(std_msgs.msg.Empty())
        rospy.sleep(0.25)'''

    def _set_motion(self, x, y, yaw, dz):
        motion = crazyflie.msg.CFMotion()
        motion.x = x
        motion.y = y
        motion.yaw = yaw
        motion.dz = dz
        motion.is_flow_motion = True
        self._ros_motion_pub.publish(motion)

    def _set_command(self, cmd):
        #0 is ESTOP, 1 IS LAND, 2 IS TAKEOFF
        command = crazyflie.msg.CFCommand()
        command.cmd = cmd
        self._ros_command_pub.publish(command)


    def _joy_cb(self, msg):
        self._curr_joy = msg

        #permanent state
        if self._curr_joy.buttons[self._joy_start_btn]:
            self._start_pressed = True

        if self._curr_joy.buttons[self._joy_stop_btn]:
            self._stop_pressed = True
        #toggle
        if self._curr_joy.buttons[self._joy_pause_run_btn]:
            self._pause = not self._pause
            self.info("Pause State Changed to: " + str(self._pause))



    def step(self, action, offline=False):


        if not offline:
            assert (self.ros_is_good())

        action = np.asarray(action)
        if not (np.logical_and(action >= self.action_space.low, action <= self.action_space.high).all()):
            logger.warn('Action {0} will be clipped to be within bounds: {1}, {2}'.format(action,
                                                                                          self.action_space.low,
                                                                                          self.action_space.high))
            action = np.clip(action, self.action_space.low, self.action_space.high)

        vx, vy, yaw, dz = action
        # print(action)
        self._set_motion(vx, vy, yaw, dz)

        if not offline:
            # print("online", self._dt, (rospy.Time.now() - self._last_step_time).to_sec())
            rospy.sleep(max(0., self._dt - (rospy.Time.now() - self._last_step_time).to_sec()))
            self._last_step_time = rospy.Time.now()

        next_observation = self._get_observation()
        goal = self._get_goal()
        reward = self._get_reward()
        done = self._get_done()

        if done:
            if self._get_collision():
                logger.warn('-- COLLISION --')
            elif self._get_joy_stop():
                logger.warn('-- MANUALLY STOPPED --')

        env_info = dict()

        self._t += 1

        if not offline:
            self._ros_rolloutbag.write_all(self._ros_topics_and_types.keys(), self._ros_msgs, self._ros_msg_times)
            if done:
                logger.debug('Done after {0} steps'.format(self._t))
                self._t = 0
                self._ros_rolloutbag.close()

        return next_observation, goal, reward, done, env_info


    def reset_state(self):
        self._last_step_time = rospy.Time.now()
        self._is_collision = False
        self._t = 0
        self._done = False
        self._curr_joy = None
        self._start_pressed = False
        self._stop_pressed = False
        self._pause = False

    def reset(self, offline=False, keep_rosbag=False):
        if offline:
            self._is_collision = False
            return self._get_observation(), self._get_goal()

        assert self.ros_is_good(), "On End: ROS IS NOT GOOD"
        #assert(not self.is_upside_down())

        
        self._set_command(0)

        if self._get_pause():
            logger.warn("Paused by User (Joystick input). Waiting...")
            while not self._get_pause():
                pass
            logger.debug("Unpaused by User (Joystick input).")

        if self._ros_rolloutbag.is_open:
            if keep_rosbag:
                self._ros_rolloutbag.close()
            else:
                # should've been closed in step when done
                logger.debug('Trashing bag')
                self._ros_rolloutbag.trash()


        #waiting for crash to be stable
        rospy.sleep(1.0)

        if self._press_enter_on_reset or self.is_upside_down():
            logger.info('Resetting, press enter to continue')
            input()
        else:
            if self._is_collision:
                logger.debug('Resetting (collision)')
            else:
                # print(self._curr_joy)
                logger.debug('Resetting (no collision)')

        
        self.reset_state()

        assert self.ros_is_good(), "On Start: ROS IS NOT GOOD"

        
        logger.debug("Waiting to takeoff ...")
        rospy.sleep(1.0)

        # for good measure
        self._is_collision = False

        logger.debug("Taking off")
        self._set_command(2)
        rospy.sleep(2.0)

        # must be after resetting curr_joy
        if self._use_joy_commands:
            logger.debug("Waiting for Start (Joystick input)...")
            while not self._get_joy_start() and not self._get_done():
                pass
            if self._get_done():
                logger.warn("* Joystick input bypassed - (likely due to collision) *")
                # reset again
                return self.reset(offline, keep_rosbag)
            self._curr_joy = None

        logger.info("Beginning Episode...")

        self._ros_rolloutbag.open()

        return self._get_observation(), self._get_goal()

    ###########
    ### ROS ###
    ###########

    def ros_msg_update(self, msg, args):
        topic = args[0]


        if 'coll' in topic:
            if msg.data == 1:
                self._is_collision = True

            if self._is_collision:
                if msg.data == 1:
                    # if is_collision and current is collision, update
                    self._ros_msgs[topic] = msg
                    self._ros_msg_times[topic] = rospy.Time.now()
                else:
                    if self._ros_msgs[topic].data != 1:
                        # if is collision, but previous message is not collision, then this topic didn't cause a colision
                        self._ros_msgs[topic] = msg
                        self._ros_msg_times[topic] = rospy.Time.now()
            else:
                # always update if not in collision
                self._ros_msgs[topic] = msg
                self._ros_msg_times[topic] = rospy.Time.now()
        else:
            #CF data unpacking
            # if 'data' in topic:
            #     self._ros_msgs['accel_x'] = msg.accel_x
            #     self._ros_msg_times['accel_x'] = rospy.Time.now()
            #     self._ros_msgs['accel_y'] = msg.accel_y
            #     self._ros_msg_times['accel_y'] = rospy.Time.now()
            #     self._ros_msgs['accel_z'] = msg.accel_z
            #     self._ros_msg_times['accel_z'] = rospy.Time.now()
            #     self._ros_msgs['alt'] = msg.alt
            #     self._ros_msg_times['alt'] = rospy.Time.now()
            #     self._ros_msgs['v_batt'] = msg.v_batt
            #     self._ros_msg_times['v_batt'] = rospy.Time.now()
            # else:
            self._ros_msgs[topic] = msg
            self._ros_msg_times[topic] = rospy.Time.now()

    def ros_is_good(self, print=True):
        # check that all not commands are coming in at a continuous rate
        # logger.debug(str(self._ros_msgs))
        # logger.debug(str(self._ros_msg_times))

        for topic in self._ros_topics_and_types.keys():
            if 'cmd' not in topic and 'coll' not in topic:
                if topic not in self._ros_msg_times:
                    if print:
                        logger.debug('Topic {0} has never been received'.format(topic))
                    return False
                elapsed = (rospy.Time.now() - self._ros_msg_times[topic]).to_sec()
                if elapsed > self._dt:
                    if print:
                        logger.debug('Topic {0} was received {1} seconds ago (dt is {2})'.format(topic, elapsed, self._dt))
                    return False

        # check if in python mode
        # if self._ros_msgs.get('mode') is None or self._ros_msgs['mode'].data != 2:
        #     if print:
        #         logger.debug('In mode {0}'.format(self._ros_msgs.get('mode')))
        #     return False

        # if 'cf/0/coll' in self._ros_msgs and self._ros_msgs['cf/0/coll'].data:
        #     if print:
        #         logger.warn('Crazyflie has crashed! Flip it if it\'s not already upright')
        #     self._is_collision = False # otherwise will stay flipped forever
        #     return False

        #TODO: Flip logic to trash experiment when crazyflie is overturn
        
        return True

    def is_upside_down(self):
        alt = self._get_observation()[1][1]
        if (alt > 0.05):
            logger.debug("Crazyflie is upside down")
            return True
        return False