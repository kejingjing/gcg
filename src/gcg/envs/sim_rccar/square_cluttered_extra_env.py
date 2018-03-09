import os

from gcg.envs.sim_rccar.square_cluttered_env import SquareClutteredEnv

class SquareClutteredExtraEnv(SquareClutteredEnv):

    def __init__(self, params={}):
        self.ignore_extra_object_collisions = params['ignore_extra_object_collisions']  # bool
        if params['use_unfamiliar_extra_objects']:
            self.extra_object_class = 'models/traffic_cone.egg'  # not seen in training data
        else:
            self.extra_object_class = 'models/chair.egg'  # seen in training data

        SquareClutteredEnv.__init__(self, params=params)

    @property
    def _model_path(self):
        return os.path.join(self._base_dir, 'square_cluttered_cone.egg')

    def _setup_map(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.extra_object_class)
        extra_object_positions = [(20., -15., 0.4),
                                  (18.3, -11.5, 0.4),
                                  (21.6, 3.9, 0.4),
                                  (20.6, 15, 0.4),
                                  (10.5, 19.5, 0.4),
                                  (10.5, 20.5, 0.4),
                                  (-4.9, 20.3, 0.4)]
        hpr = (0, 0, 0)
        scale = 0.5
        for pos in extra_object_positions:
            self._setup_collision_object(path, pos, hpr, scale=scale, ignore_collision=self.ignore_extra_object_collisions)

        self._setup_collision_object(self._model_path)


if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': False, 'hfov': 120}
    env = SquareClutteredConeEnv(params)
