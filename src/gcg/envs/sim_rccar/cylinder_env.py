import os
import numpy as np

from gcg.envs.sim_rccar.car_env import CarEnv


class CylinderEnv(CarEnv):

    def __init__(self, params={}):
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/cylinder.egg')

        CarEnv.__init__(
            self,
            params=params)

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
