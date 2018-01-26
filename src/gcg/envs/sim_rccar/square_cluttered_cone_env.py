import os, copy

from gcg.envs.sim_rccar.square_env import SquareEnv

class SquareClutteredConeEnv(SquareEnv):
    def __init__(self, params={}):
        params.setdefault('model_path', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/square_cluttered_cone.egg'))
        self.positions = None

        SquareEnv.__init__(self, params=params)

    def _setup_map(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/traffic_cone.egg')
        self.positions = [(20., -15., 0.4),
                          (18.3, -11.5, 0.4),
                          (21.6, 3.9, 0.4),
                          (20.6, 15, 0.4),
                          (10.5, 19.5, 0.4),
                          (10.5, 20.5, 0.4),
                          (-4.9, 20.3, 0.4)]
        hpr = (0, 0, 0)
        scale = 0.5
        for pos in self.positions:
            self._setup_collision_object(path, pos, hpr, scale=scale)

        self._setup_collision_object(self._model_path)

    def _default_restart_pos(self):
        l = [
            [20., -20., 0.3, 0.0, 0.0, 0.0],
            [20., -17., 0.3, 0.0, 0.0, 0.0],
            [20., -12., 0.3, 0.0, 0.0, 0.0],
            [20., -9., 0.3, 0.0, 0.0, 0.0],
            [20., -4., 0.3, 0.0, 0.0, 0.0],
            [20., 4., 0.3, 0.0, 0.0, 0.0],
            [20., 10., 0.3, 0.0, 0.0, 0.0],

            [20., 20., 0.3, 90.0, 0.0, 0.0],
            [14., 20., 0.3, 90.0, 0.0, 0.0],
            [6., 20., 0.3, 90.0, 0.0, 0.0],
            [1., 20., 0.3, 90.0, 0.0, 0.0],
            [-2., 20., 0.3, 90.0, 0.0, 0.0],
            [-8., 20., 0.3, 90.0, 0.0, 0.0],
            [-12., 20., 0.3, 90.0, 0.0, 0.0],

            [-20., 20., 0.3, 180.0, 0.0, 0.0],
            [-20., 15., 0.3, 180.0, 0.0, 0.0],
            [-20., 10., 0.3, 180.0, 0.0, 0.0],
            [-20., 5., 0.3, 180.0, 0.0, 0.0],
            [-20., 0., 0.3, 180.0, 0.0, 0.0],
            [-20., -5., 0.3, 180.0, 0.0, 0.0],
            [-20., -13., 0.3, 180.0, 0.0, 0.0],

            [-20., -20., 0.3, 270.0, 0.0, 0.0],
            [-15., -20., 0.3, 270.0, 0.0, 0.0],
            [-10., -20., 0.3, 270.0, 0.0, 0.0],
            [-5., -20., 0.3, 270.0, 0.0, 0.0],
            [-2., -20., 0.3, 270.0, 0.0, 0.0],
            [5., -20., 0.3, 270.0, 0.0, 0.0],
            [13., -20., 0.3, 270.0, 0.0, 0.0],

            ]

        for start in copy.copy(l):
            start[3] += 180.
            l.append(start)

        return l

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': False, 'hfov': 120}
    env = SquareClutteredConeEnv(params)
