import os

import numpy as np

from gcg.envs.rccar.car_env import CarEnv


class CylinderEnv(CarEnv):

    def __init__(self, params={}):
        params.setdefault('steer_limits', [-30., 30.])
        params.setdefault('speed_limits', [2., 2.])

        params.setdefault('use_depth', False)
        params.setdefault('do_back_up', False)
        self._model_path = params.get('model_path',
                                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/cylinder.egg'))

        CarEnv.__init__(
            self,
            params=params)
#        assert(self._fixed_speed)

    ### special for rllab

    def _get_observation(self):
        im = super(CylinderEnv, self)._get_observation()
        ori = self._get_heading()
        vec = np.array([ori])
        return im, vec

    def step(self, action):
        lb, ub = self._unnormalized_action_space.bounds
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        if self._fixed_speed:
            scaled_action[1] = self._speed_limits[0]

        return CarEnv.step(self, scaled_action)

    ### default

    def _default_pos(self):
        return (0.0, -6., 0.25)

    def _get_done(self):
        return self._collision or np.array(self._vehicle_pointer.getPos())[1] >= 6.0

    def _default_restart_pos(self):
        ran = np.linspace(-3.5, 3.5, 20)
        np.random.shuffle(ran)
        restart_pos = []
        for val in ran:
            restart_pos.append([val, -6.0, 0.25, 0.0, 0.0, 0.0])
        return restart_pos

    @property
    def horizon(self):
        return 24

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': False, 'hfov': 120}
    env = CylinderEnv(params)
