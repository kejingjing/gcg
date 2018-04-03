import itertools
import pickle
import numpy as np
import time 

from gcg.envs.spaces.discrete import Discrete
from gcg.envs.env_utils import create_env
from gcg.envs.spaces.box import Box
from gcg.envs.vec_env_executor import VecEnvExecutor
from gcg.sampler.replay_pool import ReplayPool
from gcg.data import mypickle


class Sampler(object):
    def __init__(self, policy, env, n_envs, replay_pool_size, max_path_length, sampling_method,
                 save_rollouts=False, save_rollouts_observations=True, save_env_infos=False, env_dict=None, replay_pool_params={}):
        self._policy = policy
        self._n_envs = n_envs

        assert(self._n_envs == 1) # b/c policy reset

        self._replay_pools = [ReplayPool(env.spec,
                                         env.horizon,
                                         policy.N,
                                         policy.gamma,
                                         replay_pool_size // n_envs,
                                         obs_history_len=policy.obs_history_len,
                                         sampling_method=sampling_method,
                                         save_rollouts=save_rollouts,
                                         save_rollouts_observations=save_rollouts_observations,
                                         save_env_infos=save_env_infos,
                                         replay_pool_params=replay_pool_params)
                              for _ in range(n_envs)]

        if self._n_envs == 1:
            envs = [env]
        else:
            try:
                envs = [pickle.loads(pickle.dumps(env)) for _ in range(self._n_envs)] if self._n_envs > 1 else [env]
            except:
                envs = [create_env(env_dict) for _ in range(self._n_envs)] if self._n_envs > 1 else [env]
        ### need to seed each environment if it is GymEnv
        # TODO: set seed
        self._vec_env = VecEnvExecutor(
            envs=envs,
            max_path_length=max_path_length
        )

        self._curr_observations, self._curr_goals = None, None

    @property
    def n_envs(self):
        return self._n_envs

    def __len__(self):
        return sum([len(rp) for rp in self._replay_pools])

    ####################
    ### Add to pools ###
    ####################

    def step(self, step, take_random_actions=False, explore=True, actions=None, **kwargs):
        """ Takes one step in each simulator and adds to respective replay pools """
        ### store last observations and get encoded
        encoded_observations_im = []
        encoded_observations_vec = []
        for i, (replay_pool, observation, goal) in enumerate(zip(self._replay_pools,
                                                                 self._curr_observations,
                                                                 self._curr_goals)):
            replay_pool.store_observation(step + i, observation, goal)
            encoded_observation = replay_pool.encode_recent_observation()
            encoded_observations_im.append(encoded_observation[0])
            encoded_observations_vec.append(encoded_observation[1])

        st = time.time()
        ### get actions
        if take_random_actions:
            assert (actions is None)
            actions = [self._vec_env.action_selection_space.sample() for _ in range(self._n_envs)]
            action_infos = [{} for _ in range(self._n_envs)]
        else:
            if actions is None:
                actions, _, action_infos = self._policy.get_actions(
                    steps=list(range(step, step + self._n_envs)),
                    current_episode_steps=self._vec_env.current_episode_steps,
                    observations=(encoded_observations_im, encoded_observations_vec),
                    goals=self._curr_goals,
                    explore=explore)
            else:
                assert (len(actions) == self.n_envs)
                action_infos = [dict()] * len(actions)

        elapsed_t = time.time() - st
        print("Elapsed in inference:", elapsed_t)


        ### take step
        next_observations, goals, rewards, dones, env_infos = self._vec_env.step(actions, **kwargs)
        for env_info, action_info in zip(env_infos, action_infos):
            env_info.update(action_info)

        if np.any(dones):
            self._policy.reset_get_action()

        ### add to replay pool
        for replay_pool, action, reward, done, env_info,  in \
                zip(self._replay_pools, actions, rewards, dones, env_infos):
            replay_pool.store_effect(action, reward, done, env_info)

        self._curr_observations = next_observations
        self._curr_goals = goals 

    def trash_current_rollouts(self):
        """ In case an error happens """
        steps_removed = 0
        for replay_pool in self._replay_pools:
            steps_removed += replay_pool.trash_current_rollout()
        return steps_removed

    def reset(self, **kwargs):
        self._curr_observations, self._curr_goals = self._vec_env.reset(**kwargs)
        for replay_pool in self._replay_pools:
            replay_pool.force_done()

    ####################
    ### Add rollouts ###
    ####################

    def add_rollouts(self, rlist, max_to_add=None):
        """
        rlist can contain pkl filenames, or dictionaries
        """
        step = sum([len(replay_pool) for replay_pool in self._replay_pools])
        replay_pools = itertools.cycle(self._replay_pools)
        done_adding = False

        for rlist_entry in rlist:
            if type(rlist_entry) is str:
                rollouts = mypickle.load(rlist_entry)['rollouts']
            elif issubclass(type(rlist_entry), dict):
                rollouts = [rlist_entry]
            else:
                raise NotImplementedError

            for rollout, replay_pool in zip(rollouts, replay_pools):
                r_len = len(rollout['dones'])
                if max_to_add is not None and step + r_len >= max_to_add:
                    diff = max_to_add - step
                    for k in ('observations', 'actions', 'rewards', 'dones'):
                        rollout[k] = rollout[k][:diff]
                    done_adding = True
                    r_len = len(rollout['dones'])

                replay_pool.store_rollout(step, rollout)
                step += r_len

                if done_adding:
                    break

            if done_adding:
                break

    #########################
    ### Sample from pools ###
    #########################

    def can_sample(self):
        return np.any([replay_pool.can_sample() for replay_pool in self._replay_pools])

    def sample(self, batch_size):
        return ReplayPool.sample_pools(self._replay_pools, batch_size,
                                       only_completed_episodes=self._policy.only_completed_episodes)

    @property
    def is_done_nexts(self):
        return self._vec_env.is_done_nexts

    ###############
    ### Logging ###
    ###############

    def log(self, prefix=''):
        ReplayPool.log_pools(self._replay_pools, prefix=prefix)

    def get_recent_paths(self):
        return ReplayPool.get_recent_paths_pools(self._replay_pools)

