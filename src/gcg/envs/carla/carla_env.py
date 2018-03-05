from collections import OrderedDict
import random

import numpy as np

from gcg.envs.env_spec import EnvSpec
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.data.logger import logger

from carla.client import make_carla_client
from carla.settings import CarlaSettings
from carla.sensor import Camera

"""
How to reset to arbitrary pose?
- cannot, must use one of preset positions
"""

class CarlaEnv:
    def __init__(self, params={}):
        # TODO: set default
        params.setdefault('host', 'localhost')
        params.setdefault('port', 2000)
        params.setdefault('cameras', ['rgb'])
        params.setdefault('camera_size', (80, 60))
        params.setdefault('rgb',
                          {
                              # for carla
                              'postprocessing': None,
                              'position': (30, 0, 130),
                              'fov': 90,

                              # for us
                              'include_in_obs': True,
                              'grayscale': True, # optional (default is False)

                          })
        params.setdefault('number_of_vehicles', 0)
        params.setdefault('number_of_pedestrians', 0)
        params.setdefault('weather', -1)

        self._carla_client, self._carla_settings, self._carla_scene = self._setup_carla(params)
        self.action_spec, self.action_selection_spec, self.observation_vec_spec, self.goal_spec, \
            self.action_space, self.action_selection_space, self.observation_im_space = self._setup_spec(params)
        self._params = params

    #############
    ### Setup ###
    #############

    def _setup_carla(self, params):
        carla_client = make_carla_client(params['host'], params['port'])

        ### create initial settings
        carla_settings = CarlaSettings()
        carla_settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=params['number_of_vehicles'],
            NumberOfPedestrians=params['number_of_pedestrians'],
            WeatherId=params['weather'])
        carla_settings.randomize_seeds()

        ### add cameras
        for camera_name in params['cameras']:
            camera_params = params['camera_name']
            camera = Camera(camera_name, PostProcessing=camera_params['postprocessing'])
            camera.set_image_size(*params['camera_size'])
            camera.set_position(*camera_params['position'])
            camera.set('CameraFOV', camera_params['fov'])
            carla_settings.add_sensor(camera)

        ### load settings
        carla_scene = carla_client.load_settings(carla_settings)

        return carla_client, carla_settings, carla_scene

    def _setup_spec(self, params):
        action_spec = OrderedDict()
        action_selection_spec = OrderedDict()
        observation_vec_spec = OrderedDict()
        goal_spec = OrderedDict()

        action_spec['steer'] = Box(low=-1., high=1.)
        action_spec['motor'] = Box(low=-1., high=1.)
        action_space = Box(low=np.array([v.low[0] for k, v in action_spec]),
                                high=np.array([v.high[0] for k, v in action_spec]))

        action_selection_spec['steer'] = Box(low=-1., high=1.)
        action_selection_spec['motor'] = Box(low=-1., high=1.)
        action_selection_space = Box(low=np.array([v.low[0] for k, v in action_selection_spec]),
                                          high=np.array([v.high[0] for k, v in action_selection_spec]))

        assert (np.logical_and(action_selection_space.low >= action_space.low,
                               action_selection_space.high <= action_space.high).all())


        num_channels = 0
        for camera_name in params['cameras']:
            camera_params = params[camera_name]
            if camera_params['include_in_obs']:
                if camera_params['postprocessing'] is None:
                    num_channels += 1 if camera_params.get('grayscale', False) else 3
                else:
                    num_channels += 1
        observation_im_space = Box(low=0, high=255, shape=list(params['camera_size']) + [num_channels])

        observation_vec_spec['collision'] = Discrete(1)
        observation_vec_spec['collision_vehicles'] = Discrete(1)
        observation_vec_spec['collision_pedestrians'] = Discrete(1)
        observation_vec_spec['collision_other'] = Discrete(1)
        observation_vec_spec['heading'] = Box(low=0., high=180.)
        observation_vec_spec['speed'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_x'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_y'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_z'] = Box(low=-100., high=100.)

        return action_spec, action_selection_spec, observation_vec_spec, goal_spec, \
                action_space, action_selection_space, observation_im_space

    ###############
    ### Get/Set ###
    ###############

    def _execute_action(self, a):
        steer = a[0]
        if a[1] > 0:
            throttle = a[1]
            brake = 0
        else:
            throttle = 0
            brake = abs(a[1])

        self._carla_client.send_control(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            reverse=False)

    def _is_collision(self, measurements, sensor_data):
        pm = measurements.player_measurements
        coll = ((pm.collision_vehicles > 0) or (pm.collision_pedestrians > 0) or (pm.collision_other > 0))
        return coll

    def _get_observation_im(self, measurements, sensor_data):
        obs_im = []

        for camera_name in self._params['cameras']:
            camera_params = self._params['camera_name']
            if camera_params['include_in_obs']:
                camera_data = sensor_data[camera_name].data
                if camera_params.get('grayscale', False):
                    def rgb2gray(rgb):
                        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
                    camera_data = np.expand_dims(rgb2gray(camera_data), 2)
                obs_im.append(camera_data)

        return np.concatenate(obs_im, axis=2)

    def _get_observation_vec(self, measurements, sensor_data):
        pm = measurements.player_measurements

        obs_vec_d = {}
        obs_vec_d['collision_vehicles'] = (pm.collision_vehicles > 0)
        obs_vec_d['collision_pedestrians'] = (pm.collision_pedestrians > 0)
        obs_vec_d['collision_other'] = (pm.collision_other > 0)
        obs_vec_d['collision'] = (obs_vec_d['collision_vehicles'] or
                                  obs_vec_d['collision_pedestrians'] or
                                  obs_vec_d['collision_other'])
        obs_vec_d['heading'] = pm.transform.rotation.yaw
        obs_vec_d['speed'] = pm.forward_speed
        obs_vec_d['accel_x'] = pm.acceleration.x
        obs_vec_d['accel_y'] = pm.acceleration.y
        obs_vec_d['accel_z'] = pm.acceleration.z

        obs_vec = np.array([obs_vec_d[k] for k in self.observation_vec_spec.keys()])
        return obs_vec

    def _get_goal(self, measurements, sensor_data):
        return np.array([])

    def _get_reward(self, measurements, sensor_data):
        if self._is_collision(measurements, sensor_data):
            reward = -1.
        else:
            reward = 0.

        return reward

    def _get_done(self, measurements, sensor_data):
        return self._is_collision(measurements, sensor_data)

    def _get_env_info(self, measurements, sensor_data):
        env_info = {}
        for camera_name in self._params['cameras']:
            camera_params = self._params['camera_name']
            if not camera_params['include_in_obs']:
                env_info[camera_name] = sensor_data[camera_name].data
        return env_info


    ######################
    ### Public methods ###
    ######################

    def step(self, action):
        self._execute_action(action)

        next_observation = (self._get_observation_im(), self._get_observation_vec())
        goal = self._get_goal()
        reward = self._get_reward()
        done = self._get_done()
        env_info = self._get_env_info()

        return next_observation, goal, reward, done, env_info

    def reset(self, player_start_idx=None):
        number_of_player_starts = len(self._carla_scene.player_start_spots)
        if player_start_idx is None:
            player_start_idx = np.random.randint(number_of_player_starts)
        else:
            player_start_idx = int(player_start_idx) % number_of_player_starts
        self._carla_client.start_episode(player_start_idx)
