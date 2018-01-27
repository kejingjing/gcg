import numpy as np

from gcg.sampler.replay_pool import ReplayPool

class BootstrapReplayPool(object):

    def __init__(self, num_bootstraps, **kwargs):
        self._num_bootstraps = num_bootstraps
        self._replay_pools = [ReplayPool(**kwargs) for _ in range(self._num_bootstraps)]

    def __len__(self):
        return len(self._replay_pools[0])

    def store_rollouts(self, start_step, rollouts):
        """
        Store rollouts by resampling with replacement
        But first replay_buffer has all data (equally)
        """
        rollouts = np.array(rollouts)
        num_rollouts = len(rollouts)

        bootstrap_indices = []
        bootstrap_indices.append(np.arange(num_rollouts))
        for b in range(self._num_bootstraps - 1):
            indices_b = np.array([np.random.randint(0, num_rollouts) for _ in range(num_rollouts)])
            bootstrap_indices.append(indices_b)

        for replay_pool, indices in zip(self._replay_pools, bootstrap_indices):
            replay_pool.store_rollouts(start_step, rollouts[indices])

    def sample(self, batch_size, include_env_infos=False, only_completed_episodes=False):
        """
        Returns batch_size * num_bootstraps
        """
        steps, observations_im, observations_vec, actions, rewards, dones, env_infos = [], [], [], [], [], [], []

        for rp in self._replay_pools:
            steps_i, (observations_im_i, observations_vec_i), actions_i, rewards_i, dones_i, env_infos_i = \
                rp.sample(batch_size,
                          include_env_infos=include_env_infos,
                          only_completed_episodes=only_completed_episodes)

            steps.append(steps_i)
            observations_im.append(observations_im_i)
            observations_vec.append(observations_vec_i)
            actions.append(actions_i)
            rewards.append(rewards_i)
            dones.append(dones_i)
            env_infos.append(env_infos_i)

        return np.concatenate(steps), (np.concatenate(observations_im), np.concatenate(observations_vec)), \
               np.concatenate(actions), np.concatenate(rewards), np.concatenate(dones), np.concatenate(env_infos)

    def sample_all_generator(self, batch_size, include_env_infos=False):
        """
        Returns all data equally (i.e. from first replay pool)
        """
        return self._replay_pools[0].sample_all_generator(batch_size, include_env_infos=include_env_infos)
