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
        # TODO maybe stagger
#        index = 0
#        max_len = len(self._obj_paths)
#        oris = [0., 90., 180., 270.]
#        for i in range(2, 27, 5):
#            for j in range(2, 27, 5):
#                if (i != 2 and i != 22) or (j!=2 and j!=22): 
#                    pos = (i - 12., j - 12., 0.3)
#                    path = self._obj_paths[index % max_len]
#                    angle = oris[(index // max_len) % 4]
#                    hpr = (angle, 0.0, 0.0)
#                    self._setup_collision_object(path, pos, hpr)
#                    index += 1
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

    def _real_reward(self):
        if self._collision:
            reward = self._collision_reward
        else:
#            reward = (np.cos(self._goal_heading[0] - self._get_heading()) + 1.) / 2.
            lb, ub = self._unnormalized_action_space.bounds
            reward = np.cos(self._goal_heading[0] - self._get_heading()) * abs(self._get_speed()) / ub[1]
#        import IPython; IPython.embed()
#        print('hi')
        assert(reward <= self.max_reward)
        return reward

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = SimpleRoomBackupEnv(params)
