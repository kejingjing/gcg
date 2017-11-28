import os

import numpy as np
import cv2

from rllab.envs.base import Env
from rllab.spaces.box import Box
from rllab.misc import logger as rllab_logger

from sandbox.gkahn.gcg.utils import logger

import rospy
import rosbag
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg

class RosMsgListener:
    def __init__(self, topic, msg_type, min_dt=None):
        self._min_dt = min_dt
        self._msg = None
        self._sub = rospy.Subscriber(topic, msg_type, self._callback)

    def get(self):
        if self._min_dt is not None and
        return self._msg

    def _callback(self, msg):
        self._msg = msg

class RWrccarEnv(Env):

    def __init__(self, params={}):
        params.setdefault('dt', 0.25)
        params.setdefault('horizon', int(5. * 60. / params['dt'])) # 5 minutes worth
        params.setdefault('ros_namespace', '/rccar/')
        params.setdefault('obs_shape', (64, 36))
        params.setdefault('steer_limits', [-1., 1.])
        params.setdefault('speed_limits', [0.3, 0.3])
        params.setdefault('collision_reward', 0.)
        params.setdefault('backup_speed', -0.2)
        params.setdefault('backup_duration', 1.)
        params.setdefault('backup_steer_range', (-0.1, 0.1))

        self._dt = params['dt']
        self.horizon = params['horizon']

        self.action_space = Box(low=np.array([params['steer_limits'][0], params['speed_limits'][0]]),
                                high=np.array([params['steer_limits'][1], params['speed_limits'][1]]))
        self.observation_space = Box(low=0, high=255, shape=params['obs_shape'])

        self._last_step_time = None
        self._is_collision = False
        self._collision_reward = params['collision_reward']
        self._backup_speed = params['backup_speed']
        self._backup_duration = params['backup_duration']
        self._backup_steer_range = params['backup_steer_range']

        ### ROS
        rospy.init_node('RWrccarEnv', anonymous=True)
        rospy.sleep(1)

        self._ros_namespace = params['ros_namespace']
        self._ros_topics_and_types = dict([
            ('camera/image_raw/compressed', sensor_msgs.msg.CompressedImage),
            ('mode', std_msgs.msg.Int32),
            ('steer', std_msgs.msg.Float32),
            ('motor', std_msgs.msg.Float32),
            ('battery/a', std_msgs.msg.Float32),
            ('battery/b', std_msgs.msg.Float32),
            ('battery/low', std_msgs.msg.Int32),
            ('encoder/left', std_msgs.msg.Float32),
            ('encoder/right', std_msgs.msg.Float32),
            ('encoder/both', std_msgs.msg.Float32),
            ('orientation/quat', geometry_msgs.msg.Quaternion),
            ('orientation/rpy', geometry_msgs.msg.Vector3),
            ('imu', geometry_msgs.msg.Accel),
            ('collision/all', std_msgs.msg.Int32),
            ('collision/flip', std_msgs.msg.Int32),
            ('collision/jolt', std_msgs.msg.Int32),
            ('collision/stuck', std_msgs.msg.Int32),
            ('cmd/steer', std_msgs.msg.Float32),
            ('cmd/motor', std_msgs.msg.Float32),
            ('cmd/vel', std_msgs.msg.Float32)
        ])
        self._ros_cb_dict = dict()
        for topic, type in self._ros_topics_and_types.items():
            rospy.Subscriber(self._ros_namespace + topic, type, (topic,))
        self._ros_steer_pub = rospy.Publisher(self._ros_namespace + 'cmd/steer', std_msgs.msg.Float32, queue_size=10)
        self._ros_vel_pub = rospy.Publisher(self._ros_namespace + 'cmd/vel', std_msgs.msg.Float32, queue_size=10)

        self._rosbag = None

        rospy.sleep(1)

    def _get_observation(self):
        ### ROS --> np
        msg = self._ros_cb_dict['camera/image_raw/compressed']
        np_arr = np.fromstring(msg.data, np.uint8)
        image_color = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR) # BGR

        ### color --> grayscale
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        image = rgb2gray(image_color)
        im = cv2.resize(image, self.observation_space.shape, interpolation=cv2.INTER_AREA)
        im = im.astype(np.uint8)

        return im

    def _get_speed(self):
        return self._ros_cb_dict['encoder/both']

    def _get_reward(self):
        reward = self._collision_reward if self._is_collision else self._get_speed()
        return reward

    def _get_done(self):
        return self._is_collision

    def _set_steer(self, steer):
        self._ros_steer_pub.publish(std_msgs.msg.Float32(steer))

    def _set_vel(self, vel):
        if self._is_collision and vel > 0:
            vel = 0.
        self._ros_vel_pub.publish(std_msgs.msg.Float32(vel))

    def step(self, action):
        lb, ub = self.action_space.bounds
        action = np.clip(action, lb, ub)

        cmd_steer, cmd_vel = action
        self._set_steer(cmd_steer)
        self._set_vel(cmd_vel)

        rospy.sleep(max(0., self._dt - (rospy.Time.now() - self._last_step_time).to_sec()))

        next_observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        env_info = dict()

        self._last_step_time = rospy.Time.now()

        return next_observation, reward, done, env_info

    def reset(self):
        if self._ros_cb_dict['collision/flip']:
            logger.warn('Car has flipped, please unflip it to continue')
            while self._ros_cb_dict['collision/flip']:
                rospy.sleep(0.1)
            logger.warn('Car is now unflipped. Continuing...')
            rospy.sleep(1.)

        backup_steer = np.random.uniform(*self._backup_steer_range)
        self._set_steer(backup_steer)
        self._set_vel(self._backup_speed)

        rospy.sleep(self._backup_duration)
        self._set_steer(0.)
        self._set_vel(0.)

        self._last_step_time = rospy.Time.now()
        self._is_collision = False

    ###########
    ### ROS ###
    ###########

    def _ros_callback(self, msg, args):
        topic = args[0]
        self._ros_cb_dict[topic] = msg

        if topic == 'collision/all':
            if msg.data == 1:
                self._is_collision = True

    @property
    def _rosbag_dir(self):
        dir = os.path.join(rllab_logger.get_snapshot_dir(), 'rosbags')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    def _rosbag_name(self, num):
        return os.path.join(self._rosbag_dir, 'rosbag{0:04d}.bag'.format(num))

    def _open_rosbag(self):
        bag_num = 0
        while os.path.exists(self._rosbag_name(bag_num)):
            bag_num += 1

        self._rosbag = rosbag.Bag(self._rosbag_name(bag_num), 'w')

    def _write_rosbag(self):
        for topic in self._ros_topics_and_types.keys():
            if topic in self._ros_cb_dict:
                self._rosbag.write(topic, self._ros_cb_dict[topic])
            else:
                logger.warn('Topic {0} not received'.format(topic))

    def _close_rosbag(self):
        self._rosbag.close()
