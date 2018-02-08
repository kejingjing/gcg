import os

from gcg.envs.sim_rccar.square_env import SquareEnv

class SquareClutteredColoredEnv(SquareEnv):
    def __init__(self, params={}):
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/square_cluttered_colored.egg')

        SquareEnv.__init__(self, params=params)

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = SquareClutteredColoredEnv(params)
