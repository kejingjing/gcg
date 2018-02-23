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

try:
    import rospy
    import rosbag
    import std_msgs.msg
    import geometry_msgs.msg
    import sensor_msgs.msg
    import crazyflie.msg
    ROS_IMPORTED = True
except:
    ROS_IMPORTED = False


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
            ('cf/0/collision', std_msgs.msg.Bool),

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
            rospy.Subscriber(self._ros_namespace + topic, type, self.ros_msg_update, (topic,))
        
        self._ros_motion_pub = rospy.Publisher("/cf/0/motion", crazyflie.msg.CFMotion)

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
        self.action_spec['yaw'] = Box(low=-1, high=1)
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

        coll = self.is_collision
        accel_x = self._ros_msgs['accel_x'].data
        accel_y = self._ros_msgs['accel_y'].data
        accel_z = self._ros_msgs['accel_z'].data
        v_batt = self._ros_msgs['v_batt'].data
        alt = self._ros_msgs['alt'].data

        vec = np.array([coll, alt, v_batt, accel_x, accel_y, accel_z])

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

    def _get_done(self):
        return self._is_collision

    def _set_steer(self, steer):
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
        rospy.sleep(0.25)

    def step(self, action, offline=False):
        if not offline:
            assert (self.ros_is_good())

        action = np.asarray(action)
        if not (np.logical_and(action >= self.action_space.low, action <= self.action_space.high).all()):
            logger.warn('Action {0} will be clipped to be within bounds: {1}, {2}'.format(action,
                                                                                          self.action_space.low,
                                                                                          self.action_space.high))
            action = np.clip(action, self.action_space.low, self.action_space.high)

        cmd_steer, cmd_vel = action
        self._set_steer(cmd_steer)
        self._set_vel(cmd_vel)

        if not offline:
            rospy.sleep(max(0., self._dt - (rospy.Time.now() - self._last_step_time).to_sec()))
            self._last_step_time = rospy.Time.now()

        next_observation = self._get_observation()
        goal = self._get_goal()
        reward = self._get_reward()
        done = self._get_done()
        env_info = dict()

        self._t += 1

        if not offline:
            self._ros_rolloutbag.write_all(self._ros_topics_and_types.keys(), self._ros_msgs, self._ros_msg_times)
            if done:
                logger.debug('Done after {0} steps'.format(self._t))
                self._t = 0
                self._ros_rolloutbag.close()

        return next_observation, goal, reward, done, env_info

    def reset(self, offline=False, keep_rosbag=False):
        if offline:
            self._is_collision = False
            return self._get_observation(), self._get_goal()

        assert (self.ros_is_good())
        
        if self._ros_rolloutbag.is_open:
            if keep_rosbag:
                self._ros_rolloutbag.close()
            else:
                # should've been closed in step when done
                logger.debug('Trashing bag')
                self._ros_rolloutbag.trash()

        if self._press_enter_on_reset:
            logger.info('Resetting, press enter to continue')
            input()
        else:
            if self._is_collision:
                logger.debug('Resetting (collision)')
            else:
                logger.debug('Resetting (no collision)')

            backup_steer = np.random.uniform(*self._backup_steer_range)
            self._set_steer(backup_steer)
            self._set_motor(self._backup_motor, self._backup_duration)
            self._set_steer(0.)
            self._set_vel(0.)

        rospy.sleep(0.5)

        self._last_step_time = rospy.Time.now()
        self._is_collision = False
        self._t = 0

        self._ros_rolloutbag.open()

        assert (self.ros_is_good())

        return self._get_observation(), self._get_goal()

    ###########
    ### ROS ###
    ###########

    def ros_msg_update(self, msg, args):
        topic = args[0]


        if 'collision' in topic:
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
            self._ros_msgs[topic] = msg
            self._ros_msg_times[topic] = rospy.Time.now()

    def ros_is_good(self, print=True):
        # check that all not commands are coming in at a continuous rate
        for topic in self._ros_topics_and_types.keys():
            if 'cmd' not in topic and 'collision' not in topic:
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
        if self._ros_msgs.get('mode') is None or self._ros_msgs['mode'].data != 2:
            if print:
                logger.debug('In mode {0}'.format(self._ros_msgs.get('mode')))
            return False

        if self._ros_msgs['collision/flip'].data:
            if print:
                logger.warn('Car has flipped, please unflip it to continue')
            self._is_collision = False # otherwise will stay flipped forever
            return False
        
        return True