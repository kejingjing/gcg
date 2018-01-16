from gcg.envs.spaces.base import Space

class EnvSpec(object):

    def __init__(
            self,
            observation_im_space,
            observation_vec_space,
            action_space):
        """
        :type observation_im_space: Space
        :type observation_vec_space: Space
        :type action_space: Space
        """
        self._observation_im_space = observation_im_space
        self._observation_vec_space = observation_vec_space
        self._action_space = action_space

    @property
    def observation_im_space(self):
        return self._observation_im_space

    @property
    def observation_vec_space(self):
        return self._observation_vec_space

    @property
    def action_space(self):
        return self._action_space
