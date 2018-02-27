import os

from gcg.envs.sim_rccar.car_env import CarEnv


class SquareEnv(CarEnv):
    @property
    def _model_path(self):
        return os.path.join(self._base_dir, 'square.egg')

    def _default_pos(self):
        return (20.0, -19., 0.3)

    def _default_restart_pos(self):
        return [
                [ 20., -20., 0.3, 0.0, 0.0, 0.0],
                [-20., -20., 0.3, 0.0, 0.0, 0.0],
                [ 20.,  15., 0.3, 0.0, 0.0, 0.0],
                [-20.,  15., 0.3, 0.0, 0.0, 0.0]
            ]

    def _get_done(self):
        return self._collision

    def reset(self, pos=None, hpr=None, hard_reset=False):
        hard_reset = hard_reset or self._env_time_step <= 2
        return CarEnv.reset(self, pos=pos, hpr=hpr, hard_reset=hard_reset)

    @property
    def horizon(self):
        # at 2m/s, roughly 80 steps per side
        return int(1e3)

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = SquareEnv(params)
