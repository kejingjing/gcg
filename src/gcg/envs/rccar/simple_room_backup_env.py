import os
from math import pi
import numpy as np

from gcg.envs.rccar.room_cluttered_env import RoomClutteredEnv

class SimpleRoomBackupEnv(RoomClutteredEnv):
    def __init__(self, params={}):
        self._base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        params.setdefault('model_path', os.path.join(self._base_dir, 'backup_room.egg'))
        params.setdefault('obj_paths', ['bookcase.egg'])
        RoomClutteredEnv.__init__(self, params=params)

    def _setup_map(self):
        self._setup_collision_object(self._model_path)

    def _default_pos(self):
        return (0.0, 0.0, 0.3)

    def _next_restart_pos_hpr_goal(self):
        pos = (0.0, 0.0, 0.3)
        goal = pi
        hpr = (0.0, 0.0, 0.0)
        return pos, hpr, goal

#    def _default_restart_pos(self):
#        return [[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]]
##        return [
##                [  11., -11., 0.3,  30.0, 0.0, 0.0], [  11., -11., 0.3,  45.0, 0.0, 0.0], [  11., -11., 0.3,  60.0, 0.0, 0.0],
##                [  11.,  11., 0.3, 120.0, 0.0, 0.0], [  11.,  11., 0.3, 135.0, 0.0, 0.0], [  11.,  11., 0.3, 150.0, 0.0, 0.0],
##                [ -11.,  11., 0.3, 210.0, 0.0, 0.0], [ -11.,  11., 0.3, 225.0, 0.0, 0.0], [ -11.,  11., 0.3, 240.0, 0.0, 0.0],
##                [ -11., -11., 0.3, 300.0, 0.0, 0.0], [ -11., -11., 0.3, 315.0, 0.0, 0.0], [ -11., -11., 0.3, 330.0, 0.0, 0.0],
##            ]

    @property
    def horizon(self):
        return 45

    def _get_reward(self):
        if self._collision:
            reward = self._collision_reward
        else:
            lb, ub = self.unnormalized_action_space.bounds
            reward = np.cos(self._goal_heading[0] - self._get_heading()) * abs(self._get_speed()) / ub[1]
        assert(reward <= self.max_reward)
        return reward

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': False, 'do_back_up': True, 'hfov': 120}
    env = SimpleRoomBackupEnv(params)
    env.step(np.array([0.0, 0.0]))
    env.reset()
