import os

from gcg.envs.sim_rccar.square_env import SquareEnv

class SquareClutteredColoredEnv(SquareEnv):
    @property
    def _model_path(self): 
        return os.path.join(self._base_dir, 'square_cluttered_colored.egg')

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = SquareClutteredColoredEnv(params)
