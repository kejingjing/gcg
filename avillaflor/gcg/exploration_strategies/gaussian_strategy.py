import numpy as np

from avillaflor.gcg.envs.spaces.box import Box
from avillaflor.gcg.utils import schedules

class GaussianStrategy(object):
    """
    Add gaussian noise
    """
    def __init__(self, env_spec, endpoints, outside_value):
        assert isinstance(env_spec.action_space, Box)
        self._env_spec = env_spec
        self.schedule = schedules.PiecewiseSchedule(endpoints=endpoints, outside_value=outside_value)

    def reset(self):
        pass

    def add_exploration(self, t, action):
        return np.clip(action + np.random.normal(size=len(action)) * self.schedule.value(t),
                       self._env_spec.action_space.low, self._env_spec.action_space.high)
