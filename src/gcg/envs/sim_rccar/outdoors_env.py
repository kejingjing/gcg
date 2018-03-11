import os, random
import numpy as np
from collections import OrderedDict

from gcg.envs.sim_rccar.room_cluttered_env import RoomClutteredEnv
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete

class OutdoorsEnv(RoomClutteredEnv):
    @property
    def _model_path(self): 
        return os.path.join(self._base_dir, 'outdoors.egg')

    def _setup_object_paths(self):
        obj_paths = ['rock4.egg', 'rock5.egg', 'tree8.egg', 'tree9.egg']
        self._obj_paths = [os.path.join(self._base_dir, x) for x in obj_paths]

    def _setup_map(self):
        self._setup_collision_object(self._model_path)

    def _default_pos(self):
        return (11.0, -11., 0.3)

    def _default_restart_pos(self):
        return [
                [  11., -11., 0.3,  30.0, 0.0, 0.0], [  11., -11., 0.3,  45.0, 0.0, 0.0], [  11., -11., 0.3,  60.0, 0.0, 0.0],
                [  11.,  11., 0.3, 120.0, 0.0, 0.0], [  11.,  11., 0.3, 135.0, 0.0, 0.0], [  11.,  11., 0.3, 150.0, 0.0, 0.0],
                [ -11.,  11., 0.3, 210.0, 0.0, 0.0], [ -11.,  11., 0.3, 225.0, 0.0, 0.0], [ -11.,  11., 0.3, 240.0, 0.0, 0.0],
                [ -11., -11., 0.3, 300.0, 0.0, 0.0], [ -11., -11., 0.3, 315.0, 0.0, 0.0], [ -11., -11., 0.3, 330.0, 0.0, 0.0],
            ]

    def _next_restart_pos_hpr_goal(self):
        num = len(self._restart_pos)
        if num == 0:
            return None, None
        else:
            pos_hpr = self._restart_pos[int(self._restart_index // 3) * 3]
            goal = self._default_restart_goal()[self._restart_index]
            self._restart_index = (self._restart_index + 1) % num
            return pos_hpr[:3], pos_hpr[3:], goal

    def _setup_spec(self):
        RoomClutteredEnv._setup_spec(self)
        self.observation_vec_spec = OrderedDict()
        self.goal_spec = OrderedDict()
        self.observation_vec_spec['coll'] = Discrete(1)
        self.observation_vec_spec['head_x'] = Box(low=-1., high=1.)
        self.observation_vec_spec['head_y'] = Box(low=-1., high=1.)
        self.observation_vec_spec['speed'] = Box(low=-4.0, high=4.0)
        self.goal_spec['speed'] = Box(low=-4.0, high=4.0)
#        self.goal_spec['heading'] = Box(low=0, high=2 * 3.14)
        self.goal_spec['head_x'] = Box(low=-1., high=1.) 
        self.goal_spec['head_y'] = Box(low=-1., high=1.) 

    def _get_observation(self):
        obs_im, obs_vec = RoomClutteredEnv._get_observation(self)
        heading = obs_vec[1]
        head_x = np.cos(heading)
        head_y = np.sin(heading)
        new_obs_vec = np.array([obs_vec[0], head_x, head_y, obs_vec[2]])
        return obs_im, new_obs_vec
    
    def _get_goal(self):
        heading = self._goal_heading
        head_x = np.cos(heading)
        head_y = np.sin(heading)
        goal = np.array([self._goal_speed, head_x, head_y])
        return goal
    
    @property
    def horizon(self):
        return 45

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = ForestEnv(params)
