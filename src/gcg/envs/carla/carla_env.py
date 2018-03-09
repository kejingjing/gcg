import os
from collections import OrderedDict
import subprocess
import random
import signal
import time

import numpy as np

from gcg.envs.env_spec import EnvSpec
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.data.logger import logger

from carla.client import CarlaClient
from carla.settings import CarlaSettings
from carla.sensor import Camera

"""
How to reset to arbitrary pose?
- cannot, must use one of preset positions
"""

class CarlaEnv:
    RETRIES = 4

    def __init__(self, params={}):
        params.setdefault('server_bash_path', '~/source/carla/CarlaUE4.sh')
        params.setdefault('map', '/Game/Maps/Town01')
        params.setdefault('fps', 4)
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

        params.setdefault('player_start_indices', None)
        params.setdefault('horizon', 1000)
        params.setdefault('goal_speed', 25) # km/hr

        self._player_start_indices = params['player_start_indices']
        self.horizon = params['horizon']
        self._goal_speed = params['goal_speed']

        self._params = params
        self._carla_server_process = None
        self._carla_client = None
        self._carla_settings = None
        self._carla_scene = None
        self.action_spec = None
        self.action_selection_spec = None
        self.observation_vec_spec = None
        self.goal_spec = None
        self.action_space = None
        self.action_selection_space = None
        self.observation_im_space = None
        self.spec = None
        self._last_measurements = None
        self._last_sensor_data = None

        ### setup
        self._setup_carla()
        self._setup_spec()

    #############
    ### Setup ###
    #############

    def _clear_carla_server(self):
        try:
            if self._carla_client is not None:
                self._carla_client.disconnect()
        except Exception as e:
            logger.warn('Error disconnecting client: {}'.format(e))
        self._carla_client = None

        if self._carla_server_process:
            pgid = os.getpgid(self._carla_server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self._carla_server_process = None

    def _setup_carla_server(self):
        assert (self._carla_server_process is None)

        server_bash_path = os.path.abspath(os.path.expanduser(self._params['server_bash_path']))
        map = self._params['map']
        fps = self._params['fps']
        port = self._params['port']

        # carla_settings_path = '/home/gkahn/source/carla/CarlaSettings.ini'
        carla_settings_path = '/tmp/CarlaSettings.ini'
        with open(carla_settings_path, 'w') as f:
            f.writelines(['[CARLA/Server]\n', 'UseNetworking=true\n', 'ServerTimeOut=1000000\n'])
        carla_settings_path = os.path.relpath(carla_settings_path, os.path.abspath(__file__))

        assert((map == '/Game/Maps/Town01') or (map == '/Game/Maps/Town02'))

        # kill_cmd = 'pkill -9 -f {0}'.format(server_bash_path)
        # subprocess.Popen(kill_cmd, shell=True)

        # cmd = 'bash {0} {1} -carla-server -benchmark -fps={2} -carla-world-port={3} -carla-settings="{4}" -carla-no-hud -windowed -ResX=400 -ResY=300'.format(server_bash_path, map, fps, port, carla_settings_path)
        cmd = [server_bash_path,
               map,
               '-carla-server',
               '-benchmark',
               '-fps={0}'.format(fps),
               '-carla-settings="{0}"'.format(carla_settings_path),
               '-carla-world-port={0}'.format(port),
               '-carla-no-hud',
               '-windowed',
               '-ResX=400',
               '-ResY=300'
               ]
        print(' '.join(cmd))
        self._carla_server_process = subprocess.Popen(cmd,
                                                      stdout=open(os.devnull, 'w'),
                                                      preexec_fn=os.setsid)

        # time.sleep(2)

    def _setup_carla_client(self):
        carla_client = CarlaClient(self._params['host'], self._params['port'], timeout=None)
        carla_client.connect()

        ### create initial settings
        carla_settings = CarlaSettings()
        carla_settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=self._params['number_of_vehicles'],
            NumberOfPedestrians=self._params['number_of_pedestrians'],
            WeatherId=self._params['weather'])
        carla_settings.randomize_seeds()

        ### add cameras
        for camera_name in self._params['cameras']:
            camera_params = self._params[camera_name]
            camera = Camera(camera_name, PostProcessing=camera_params['postprocessing'])
            camera.set_image_size(*self._params['camera_size'])
            camera.set_position(*camera_params['position'])
            camera.set(**{'CameraFOV': camera_params['fov']})
            carla_settings.add_sensor(camera)

        ### load settings
        carla_scene = carla_client.load_settings(carla_settings)

        self._carla_client = carla_client
        self._carla_settings = carla_settings
        self._carla_scene = carla_scene

    def _setup_carla(self):
        self._clear_carla_server()
        self._setup_carla_server()
        self._setup_carla_client()

    def _setup_spec(self):
        action_spec = OrderedDict()
        action_selection_spec = OrderedDict()
        observation_vec_spec = OrderedDict()
        goal_spec = OrderedDict()

        action_spec['steer'] = Box(low=-1., high=1.)
        action_spec['motor'] = Box(low=-1., high=1.)
        action_space = Box(low=np.array([v.low[0] for k, v in action_spec.items()]),
                                high=np.array([v.high[0] for k, v in action_spec.items()]))

        action_selection_spec['steer'] = Box(low=-1., high=1.)
        action_selection_spec['motor'] = Box(low=-0.5, high=1.)
        action_selection_space = Box(low=np.array([v.low[0] for k, v in action_selection_spec.items()]),
                                          high=np.array([v.high[0] for k, v in action_selection_spec.items()]))

        assert (np.logical_and(action_selection_space.low >= action_space.low,
                               action_selection_space.high <= action_space.high).all())


        num_channels = 0
        for camera_name in self._params['cameras']:
            camera_params = self._params[camera_name]
            if camera_params['include_in_obs']:
                if camera_params['postprocessing'] is None:
                    num_channels += 1 if camera_params.get('grayscale', False) else 3
                else:
                    num_channels += 1
        observation_im_space = Box(low=0, high=255, shape=list(self._params['camera_size']) + [num_channels])

        observation_vec_spec['coll'] = Discrete(1)
        observation_vec_spec['coll_car'] = Discrete(1)
        observation_vec_spec['coll_ped'] = Discrete(1)
        observation_vec_spec['coll_oth'] = Discrete(1)
        observation_vec_spec['heading'] = Box(low=0., high=180.)
        observation_vec_spec['speed'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_x'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_y'] = Box(low=-100., high=100.)
        observation_vec_spec['accel_z'] = Box(low=-100., high=100.)

        goal_spec['speed'] = Box(low=-50.0, high=50.0)

        self.action_spec, self.action_selection_spec, self.observation_vec_spec, self.goal_spec, \
        self.action_space, self.action_selection_space, self.observation_im_space = \
            action_spec, action_selection_spec, observation_vec_spec, goal_spec, \
                action_space, action_selection_space, observation_im_space

        self.spec = EnvSpec(
            observation_im_space=observation_im_space,
            action_space=action_space,
            action_selection_space=action_selection_space,
            observation_vec_spec=observation_vec_spec,
            action_spec=action_spec,
            action_selection_spec=action_selection_spec,
            goal_spec=goal_spec)

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
            camera_params = self._params[camera_name]
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
        obs_vec_d['coll_car'] = (pm.collision_vehicles > 0)
        obs_vec_d['coll_ped'] = (pm.collision_pedestrians > 0)
        obs_vec_d['coll_oth'] = (pm.collision_other > 0)
        obs_vec_d['coll'] = (obs_vec_d['coll_car'] or
                             obs_vec_d['coll_ped'] or
                             obs_vec_d['coll_oth'])
        obs_vec_d['heading'] = pm.transform.rotation.yaw
        obs_vec_d['speed'] = pm.forward_speed
        obs_vec_d['accel_x'] = pm.acceleration.x
        obs_vec_d['accel_y'] = pm.acceleration.y
        obs_vec_d['accel_z'] = pm.acceleration.z

        obs_vec = np.array([obs_vec_d[k] for k in self.observation_vec_spec.keys()])
        return obs_vec

    def _get_goal(self, measurements, sensor_data):
        return np.array([self._goal_speed])

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
            camera_params = self._params[camera_name]
            if not camera_params['include_in_obs']:
                env_info[camera_name] = sensor_data[camera_name].data
        return env_info

    ######################
    ### Public methods ###
    ######################

    def _get(self, measurements=None, sensor_data=None):
        if measurements is None or sensor_data is None:
            measurements, sensor_data = self._carla_client.read_data()

        self._last_measurements = measurements
        self._last_sensor_data = sensor_data

        next_observation = (self._get_observation_im(measurements, sensor_data),
                            self._get_observation_vec(measurements, sensor_data))
        goal = self._get_goal(measurements, sensor_data)
        reward = self._get_reward(measurements, sensor_data)
        done = self._get_done(measurements, sensor_data)
        env_info = self._get_env_info(measurements, sensor_data)

        return next_observation, goal, reward, done, env_info


    def step(self, action):
        try:
            self._execute_action(action)
            next_observation, goal, reward, done, env_info = self._get()
        except Exception as e:
            logger.warn('CarlaEnv: Error during step: {}'.format(e))
            self._setup_carla()
            next_observation, goal, reward, done, env_info = self._get(measurements=self._last_measurements,
                                                                       sensor_data=self._last_sensor_data)
            done = True

        return next_observation, goal, reward, done, env_info

    def reset(self, player_start_idx=None):
        number_of_player_starts = len(self._carla_scene.player_start_spots)
        if player_start_idx is None:
            if self._player_start_indices is None:
                player_start_idx = np.random.randint(number_of_player_starts)
            else:
                player_start_idx = random.choice(self._player_start_indices)
        else:
            player_start_idx = int(player_start_idx) % number_of_player_starts
        assert ((0 <= player_start_idx) and (player_start_idx < number_of_player_starts))

        error = None
        for _ in range (CarlaEnv.RETRIES):
            try:
                self._carla_client.start_episode(player_start_idx)
                next_observation, goal, reward, done, env_info = self._get()
                return next_observation, goal
            except Exception as e:
                logger.warn('CarlaEnv: start episode error: {}'.format(e))
                self._setup_carla()
                error = e
        else:
            logger.critical('CarlaEnv: Failed to restart after {0} attempts'.format(CarlaEnv.RETRIES))
            raise error

if __name__ == '__main__':
    logger.setup(display_name='CarlaEnv', log_path='/tmp/log.txt', lvl='debug')
    env = CarlaEnv()
    import IPython; IPython.embed()