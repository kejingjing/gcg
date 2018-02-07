import time
import itertools
from collections import defaultdict
import numpy as np

from gcg.data.logger import logger

class ReplayPool(object):

    def __init__(self, env_spec, env_horizon, N, gamma, size, obs_history_len, sampling_method,
                 save_rollouts=False, save_rollouts_observations=True, save_env_infos=False, replay_pool_params={}):
        """
        :param env_spec: for observation/action dimensions
        :param N: horizon length
        :param gamma: discount factor
        :param size: size of pool
        :param obs_history_len: how many previous obs to include when sampling? (= 1 is only current observation)
        :param sampling_method: how to sample the replay pool
        :param save_rollouts: for debugging
        """
        self._env_spec = env_spec
        self._env_horizon = env_horizon
        self._N = N
        self._gamma = gamma
        self._size = int(size)
        self._obs_history_len = obs_history_len
        self._sampling_method = sampling_method
        self._save_rollouts = save_rollouts
        self._save_rollouts_observations = save_rollouts_observations
        self._save_env_infos = save_env_infos
        self._replay_pool_params = replay_pool_params # TODO: hack

        ### buffer
        self._obs_im_shape = self._env_spec.observation_im_space.shape
        self._obs_im_dim = np.prod(self._obs_im_shape)
        self._obs_vec_dim = len(list(self._env_spec.observation_vec_spec.keys())) 
        self._action_dim = self._env_spec.action_space.flat_dim
        self._steps = np.empty((self._size,), dtype=np.int32)
        self._observations_im = np.empty((self._size, self._obs_im_dim), dtype=np.uint8)
        self._observations_vec = np.ones((self._size, self._obs_vec_dim), dtype=np.float32)
        self._actions = np.nan * np.ones((self._size, self._action_dim), dtype=np.float32)
        self._rewards = np.nan * np.ones((self._size,), dtype=np.float32)
        self._dones = np.zeros((self._size,), dtype=bool) # initialize as all not done
        self._env_infos = np.empty((self._size,), dtype=np.object)
        self._sampling_indices = np.zeros((self._size,), dtype=bool)
        self._index = 0
        self._curr_size = 0
        self._done_indices = []

        ### keep track of statistics
        self._stats = defaultdict(int)
        self._obs_im_mean = (0.5 * 255) * np.ones((1, self._obs_im_dim))
        self._obs_im_orth = (0.5 * 255) * np.ones(self._obs_im_dim)

        ### logging
        self._last_done_index = 0
        self._log_stats = defaultdict(list)
        self._log_paths = []
        self._last_get_log_stats_time = None

    def __len__(self):
        return self._curr_size

    def _get_indices(self, start, end):
        if start <= end:
            return list(range(start, end))
        elif start > end:
            return list(range(start, len(self))) + list(range(end))

    def _get_prev_indices(self, end, length):
        if end - (length - 1) >= 0:
            preclipped_indices = list(range(end - length + 1, end + 1))
        else:
            if len(self) == self._size:
                preclipped_indices = list(range(len(self) - (length - 1 - end), len(self))) + list(range(0, end+1))
            else:
                preclipped_indices = list(range(0, end+1))

        indices = [preclipped_indices[-1]]
        for index in preclipped_indices[-2::-1]:
            if self._dones[index]:
                break
            indices.insert(0, index)
        return indices

    ###################
    ### Add to pool ###
    ###################

    def store_observation(self, step, observation):
        self._steps[self._index] = step
        self._observations_im[self._index, :] = observation[0].reshape((self._obs_im_dim,)) #self._env_spec.observation_im_space.flatten(observation)
        self._observations_vec[self._index, :] = observation[1].reshape((self._obs_vec_dim,)) #self._env_spec.observation_vec__space.flatten(observation)

    def _encode_observation(self, index):
        """ Encodes observation starting at index by concatenating obs_history_len previous """
        indices = self._get_indices(index - self._obs_history_len + 1, index + 1)  # plus 1 b/c inclusive
        observations_im = self._observations_im[indices]
        observations_vec = self._observations_vec[indices]
        dones = self._dones[indices]

        encountered_done = False
        for i in range(len(dones) - 2, -1, -1):  # skip the most current frame since don't know if it's done
            encountered_done = encountered_done or dones[i]
            if encountered_done:
                observations_im[i, ...] = 0.
                observations_vec[i, ...] = 0.

        return observations_im, observations_vec

    def encode_recent_observation(self):
        return self._encode_observation(self._index)

    def _done_update(self, update_log_stats=True):
        if self._last_done_index == self._index:
            return

        self._done_indices.append(self._index - 1)
        indices = self._get_indices(self._last_done_index, self._index)
        rewards = self._rewards[indices]

        ### update log stats
        if update_log_stats:
            self._update_log_stats()

        self._last_done_index = self._index

    def store_effect(self, action, reward, done, env_info, flatten_action=True, update_log_stats=True):
        self._actions[self._index, :] = self._env_spec.action_space.flatten(action) if flatten_action else action
        self._rewards[self._index] = reward
        if self._dones[self._index] and not done:
            assert(self._curr_size + 1 >= self._size)
            if self._index in self._done_indices:
                self._done_indices.remove(self._index)

        self._dones[self._index] = done
        self._env_infos[self._index] = env_info if self._save_env_infos else None
        if self._sampling_method == 'uniform':
            pass
        elif self._sampling_method == 'nonzero':
            curr_start_indices = self._get_prev_indices(self._index, self._N)
            if reward != 0:
                self._sampling_indices[curr_start_indices] = True
            elif len(self) > self._N:
                prev_start_indices = self._get_prev_indices(self._index - 1, self._N)
                if len(prev_start_indices) > 0 and \
                   np.all(self._rewards[prev_start_indices] == 0):
                    self._sampling_indices[prev_start_indices[0]] = False
        elif self._sampling_method == 'terminal':
            start_indices = self._get_prev_indices(self._index, self._N)
            if done:
                if len(self._get_indices(self._last_done_index, self._index)) == self._env_horizon - 1:
                    self._sampling_indices[start_indices] = False
                else:
                    self._sampling_indices[start_indices] = True
            else:
                self._sampling_indices[start_indices[0]] = False
        else:
            raise NotImplementedError
        self._curr_size = max(self._curr_size, self._index + 1)
        self._index = (self._index + 1) % self._size

        ### compute values
        if done:
            self._done_update(update_log_stats=update_log_stats)

    def force_done(self):
        if len(self) == 0:
            return

        self._dones[(self._index - 1) % self._size] = True
        self._done_update()

    def store_rollout(self, start_step, rollout):
        if not rollout['dones'][-1]:
            # import IPython; IPython.embed()
            print('NOT ENDING IN DONE')
            return
        # assert(rollout['dones'][-1])

        r_len = len(rollout['dones'])
        for i in range(r_len):
            self.store_observation(start_step + i, (rollout['observations_im'][i], rollout['observations_vec'][i]))
            self.store_effect(rollout['actions'][i],
                              rollout['rewards'][i],
                              rollout['dones'][i],
                              rollout['env_infos'][i],
                              update_log_stats=False)

    def store_rollouts(self, start_step, rollouts):
        step = start_step
        for rollout in rollouts:
            self.store_rollout(step, rollout)
            step += len(rollout['dones'])

    ########################
    ### Remove from pool ###
    ########################

    def trash_current_rollout(self):
        self._actions[self._last_done_index:self._index, :] = np.nan
        self._rewards[self._last_done_index:self._index] = np.nan
        self._dones[self._last_done_index:self._index] = False
        self._env_infos[self._last_done_index:self._index] = np.object
        self._sampling_indices[self._last_done_index:self._index] = False

        r_len = (self._index - self._last_done_index) % self._size
        if self._curr_size < self._size - 1: # TODO: not sure this is right
            self._curr_size -= r_len
        
        for index in range(self._last_done_index, self._index):
            if index in self._done_indices:
                self._done_indices.remove(index)
        
        self._index = self._last_done_index


        return r_len

    ########################
    ### Sample from pool ###
    ########################

    def can_sample(self):
        return len(self) > self._obs_history_len and len(self) > self._N

    def _sample_start_indices(self, batch_size, only_completed_episodes):
        start_indices = []
        false_indices = self._get_indices(self._index - self._obs_history_len, self._index + self._N)
        false_indices += self._done_indices
        if only_completed_episodes and self._last_done_index != self._index:
            false_indices += self._get_indices(self._last_done_index, self._index)

        if self._sampling_method == 'uniform':
            hits = 0.
            tries= 0.
            while len(start_indices) < batch_size:
                start_index = np.random.randint(low=0, high=len(self) - self._N)
                if start_index not in false_indices:
                    start_indices.append(start_index)
        elif self._sampling_method == 'nonzero' or self._sampling_method == 'terminal':
            nonzero_indices = np.nonzero(self._sampling_indices[:len(self) - self._N])[0] # terminal
            zero_indices = np.nonzero(self._sampling_indices[:len(self) - self._N] == 0)[0]
            frac_terminal = self._replay_pool_params['terminal']['frac']

            while len(start_indices) < batch_size:
                if len(nonzero_indices) == 0 or len(zero_indices) == 0:
                    start_index = np.random.randint(low=0, high=len(self) - self._N)
                else:
                    if len(start_indices) < frac_terminal * batch_size:
                        start_index = np.random.choice(nonzero_indices)
                    else:
                        start_index = np.random.choice(zero_indices)

                if start_index not in false_indices:
                    start_indices.append(start_index)
        else:
            raise NotImplementedError

        return start_indices

    def sample(self, batch_size, include_env_infos=False, only_completed_episodes=False):
        """
        :return observations, actions, and rewards of horizon H+1
        """
        if not self.can_sample():
            return None

        steps, observations_im, observations_vec, actions, rewards, dones, env_infos = [], [], [], [], [], [], []
        
        start_indices = self._sample_start_indices(batch_size, only_completed_episodes)

        for start_index in start_indices:
            indices = self._get_indices(start_index, (start_index + self._N + 1) % self._curr_size)
            steps_i = self._steps[indices]
            obs_im_i, obs_vec_i = self._encode_observation(start_index)
            observations_im_i = np.vstack([obs_im_i, self._observations_im[indices[1:]]])
            observations_vec_i = np.vstack([obs_vec_i, self._observations_vec[indices[1:]]])
            actions_i = self._actions[indices]
            rewards_i = self._rewards[indices]
            dones_i = self._dones[indices]
            env_infos_i = self._env_infos[indices] if include_env_infos else [None] * len(dones_i)
            if dones_i[0]:
                raise Exception('Should not ever happen')
            if np.any(dones_i[:-1]):
                # H = 3
                # observations = [0 1 2 3]
                # actions = [10 11 12 13]
                # rewards = [20 21 22 23]
                # dones = [False True False False]

                d_idx = np.argmax(dones_i)
                # TODO maybe wrong
                for j in range(d_idx, len(dones_i)):
                    actions_i[j, :] = self._env_spec.action_space.flatten(self._env_spec.action_space.sample())
                    rewards_i[j] = 0.
                    dones_i[j] = True

                # observations = [0 1 2 3]
                # actions = [10 11 rand rand]
                # rewards = [20 21 0 0]
                # dones = [False True True True]

            steps.append(np.expand_dims(steps_i, 0))
            observations_im.append(np.expand_dims(observations_im_i, 0))
            observations_vec.append(np.expand_dims(observations_vec_i, 0))
            actions.append(np.expand_dims(actions_i, 0))
            rewards.append(np.expand_dims(rewards_i, 0))
            dones.append(np.expand_dims(dones_i, 0))
            env_infos.append(np.expand_dims(env_infos_i, 0))

        steps = np.vstack(steps)
        observations_im = np.vstack(observations_im)
        observations_vec = np.vstack(observations_vec)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)
        env_infos = np.vstack(env_infos)

        return steps, (observations_im, observations_vec), actions, rewards, dones, env_infos

    def sample_all_generator(self, batch_size, include_env_infos=False):
        if not self.can_sample():
            return

        steps, observations_im, observations_vec, actions, rewards, dones, env_infos = [], [], [], [], [], [], []

        for start_index in range(len(self) - self._N):
            indices = self._get_indices(start_index, (start_index + self._N + 1) % self._curr_size)

            steps_i = self._steps[indices]
            obs_im_i, obs_vec_i = self._encode_observation(start_index)
            observations_im_i = np.vstack([obs_im_i, self._observations_im[indices[1:]]])
            observations_vec_i = np.vstack([obs_vec_i, self._observations_vec[indices[1:]]])
            actions_i = self._actions[indices]
            rewards_i = self._rewards[indices]
            dones_i = self._dones[indices]
            env_infos_i = self._env_infos[indices] if include_env_infos else [None] * len(dones_i)

            if dones_i[0]:
                continue

            steps.append(np.expand_dims(steps_i, 0))
            observations_im.append(np.expand_dims(observations_im_i, 0))
            observations_vec.append(np.expand_dims(observations_vec_i, 0))
            actions.append(np.expand_dims(actions_i, 0))
            rewards.append(np.expand_dims(rewards_i, 0))
            dones.append(np.expand_dims(dones_i, 0))
            env_infos.append(np.expand_dims(env_infos_i, 0))

            if len(steps) >= batch_size:
                yield np.vstack(steps), (np.vstack(observations_im), np.vstack(observations_vec)),\
                      np.vstack(actions), np.vstack(rewards), np.vstack(dones), np.vstack(env_infos)
                steps, observations_im, observations_vec, actions, rewards, dones, env_infos = [], [], [], [], [], [], []

        yield np.vstack(steps), (np.vstack(observations_im), np.vstack(observations_vec)), \
              np.vstack(actions), np.vstack(rewards), np.vstack(dones), np.vstack(env_infos)

    @staticmethod
    def sample_pools(replay_pools, batch_size, include_env_infos=False, only_completed_episodes=False):
        """ Sample from replay pools (treating them as one big replay pool) """
        if not np.any([replay_pool.can_sample() for replay_pool in replay_pools]):
            return None

        steps, observations_im, observations_vec, actions, rewards, dones, env_infos = [], [], [], [], [], [], []
        
        # calculate ratio of pool sizes
        pool_lens = np.array([replay_pool.can_sample() * len(replay_pool) for replay_pool in replay_pools]).astype(float)
        pool_ratios = pool_lens / pool_lens.sum()
        # how many from each pool
        choices = np.random.choice(range(len(replay_pools)), size=batch_size, p=pool_ratios)
        batch_sizes = np.bincount(choices, minlength=len(replay_pools))
        # sample from each pool
        for i, replay_pool in enumerate(replay_pools):
            if batch_sizes[i] == 0:
                continue

            steps_i, (observations_im_i, observations_vec_i), actions_i, rewards_i, dones_i, env_infos_i = \
                replay_pool.sample(batch_sizes[i], include_env_infos=include_env_infos,
                                   only_completed_episodes=only_completed_episodes)
            steps.append(steps_i)
            observations_im.append(observations_im_i)
            observations_vec.append(observations_vec_i)
            actions.append(actions_i)
            rewards.append(rewards_i)
            dones.append(dones_i)
            env_infos.append(env_infos_i)

        steps = np.vstack(steps)
        observations_im = np.vstack(observations_im)
        observations_vec = np.vstack(observations_vec)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)
        env_infos = np.vstack(env_infos)

        for arr in (steps, observations_im, observations_vec, actions, rewards, dones, env_infos):
            assert(len(arr) == batch_size)

        return steps, (observations_im, observations_vec), actions, rewards, dones, env_infos

    ###############
    ### Logging ###
    ###############

    def _update_log_stats(self):
        indices = self._get_indices(self._last_done_index, self._index)

        ### update log
        rewards = self._rewards[indices]
        obs_vec = self._observations_vec[indices]
        coll_index = list(self._env_spec.observation_vec_spec.keys()).index('coll')
        self._log_stats['AvgCollision'].append(obs_vec[-1, coll_index])
        self._log_stats['FinalReward'].append(rewards[-1])
        self._log_stats['AvgReward'].append(np.mean(rewards))
        self._log_stats['CumReward'].append(np.sum(rewards))
        self._log_stats['EpisodeLength'].append(len(rewards))

        ## update paths
        if self._save_rollouts:
            self._log_paths.append({
                'steps': self._steps[indices],
                'observations_im': self._observations_im[indices] if self._save_rollouts_observations else None,
                'observations_vec': self._observations_vec[indices] if self._save_rollouts_observations else None,
                'actions': self._actions[indices],
                'rewards': self._rewards[indices],
                'dones': self._dones[indices],
                'env_infos': self._env_infos[indices],
            })

    def get_log_stats(self):
        self._log_stats['Time'] = [time.time() - self._last_get_log_stats_time] if self._last_get_log_stats_time else [0.]
        d = self._log_stats
        self._last_get_log_stats_time = time.time()
        self._log_stats = defaultdict(list)
        return d

    def get_recent_paths(self):
        paths = self._log_paths
        self._log_paths = []
        return paths

    @staticmethod
    def log_pools(replay_pools, prefix=''):
        def join(l):
            return list(itertools.chain(*l))
        all_log_stats = [replay_pool.get_log_stats() for replay_pool in replay_pools]
        log_stats = defaultdict(list)
        for k in all_log_stats[0].keys():
            log_stats[k] = join([ls[k] for ls in all_log_stats])
        logger.record_tabular(prefix+'CumRewardMean', np.mean(log_stats['CumReward']))
        logger.record_tabular(prefix+'CumRewardStd', np.std(log_stats['CumReward']))
        logger.record_tabular(prefix+'AvgRewardMean', np.mean(log_stats['AvgReward']))
        logger.record_tabular(prefix+'AvgRewardStd', np.std(log_stats['AvgReward']))
        logger.record_tabular(prefix+'FinalRewardMean', np.mean(log_stats['FinalReward']))
        logger.record_tabular(prefix+'FinalRewardStd', np.std(log_stats['FinalReward']))
        logger.record_tabular(prefix+'EpisodeLengthMean', np.mean(log_stats['EpisodeLength']))
        logger.record_tabular(prefix+'EpisodeLengthStd', np.std(log_stats['EpisodeLength']))

        logger.record_tabular(prefix+'AvgCollision', np.mean(log_stats['AvgCollision']))
        logger.record_tabular(prefix+'NumEpisodes', len(log_stats['EpisodeLength']))
        logger.record_tabular(prefix+'Time', np.mean(log_stats['Time']))

    @staticmethod
    def get_recent_paths_pools(replay_pools):
        return list(itertools.chain(*[rp.get_recent_paths() for rp in replay_pools]))
