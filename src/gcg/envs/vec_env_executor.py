import copy
import numpy as np


class VecEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._action_selection_space = envs[0].action_selection_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length
        self._skips = np.array([False] * len(self.envs))

    def step(self, action_n, **kwargs):
        all_results = []
        for i, env in enumerate(self.envs):
            if self._skips[i]:
                ob, goal = env.reset(**kwargs)
                self.ts[i] = 0
                all_result = [ob, goal, 0., True, {}]
                all_results.append(all_result)
            else:
                all_result = env.step(action_n[i], **kwargs)
                all_results.append(all_result)
                self.ts[i] += 1
        obs, goals, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if self._skips[i]:
                self._skips[i] = False
            elif done:
                self._skips[i] = True
                dones[i] = False
        return obs, goals, rewards, dones, env_infos

    def reset(self, **kwargs):
        obs = []
        goals = []
        for env in self.envs:
            ob, goal = env.reset(**kwargs)
            obs.append(ob)
            goals.append(goal)
        self.ts[:] = 0
        self._skips[:] = False
        return obs, goals

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def action_selection_space(self):
        return self._action_selection_space

    def terminate(self):
        pass

    @property
    def current_episode_steps(self):
        return np.copy(self.ts)

    @property
    def is_done_nexts(self):
        return copy.copy(self._skips)
