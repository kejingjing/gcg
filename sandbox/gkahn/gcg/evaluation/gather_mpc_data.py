import argparse
import random

import numpy as np

from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.evaluation.gather_random_data import GatherRandomData
from sandbox.gkahn.gcg.utils import mypickle
from sandbox.gkahn.gcg.sampler.replay_pool import ReplayPool
from sandbox.gkahn.gcg.utils.utils import inner_env

class GatherMPCData(object):
    def __init__(self, env, steps, save_file, H, K):
        self._env = env
        self._inner_env = inner_env(env)
        self._steps = steps
        self._save_file = save_file
        self._H = H
        self._K = K

        # get starting positions
        grd = GatherRandomData(env, int(100*steps), '/tmp/tmp.pkl')
        grd.run()

        rollouts = mypickle.load('/tmp/tmp.pkl')['rollouts']
        self._start_poses = []
        while len(self._start_poses) < steps:
            rollout = random.choice(rollouts)
            env_info = random.choice(rollout['env_infos'])
            self._start_poses.append((env_info['pos'], env_info['hpr']))

        self._replay_pool = ReplayPool(self._env.spec,
                                       self._env.horizon,
                                       1,
                                       1,
                                       int(1e5),
                                       obs_history_len=1,
                                       sampling_method='uniform',
                                       save_rollouts=True,
                                       save_rollouts_observations=True,
                                       save_env_infos=True,
                                       replay_pool_params={})

        self._action_sequences = []
        for _ in range(self._K):
            self._action_sequences.append([self._env.action_space.sample() for _ in range(self._H)])
        self._action_sequences = np.array(self._action_sequences)

    def run(self):
        rollouts = []
        step = 0
        for start_pose in self._start_poses:
            if len(rollouts) > self._steps:
                break

            rollouts_i = []
            for action_sequence in self._action_sequences:
                curr_obs = self._inner_env.reset(start_pose[0], start_pose[1])
                for action in action_sequence:
                    self._replay_pool.store_observation(step, curr_obs)
                    next_obs, reward, done, env_info = self._env.step(action)
                    self._replay_pool.store_effect(action, reward, done, env_info, np.nan, np.nan)
                    step += 1
                    if done:
                        break
                    curr_obs = next_obs
                else:
                    self._replay_pool.force_done()

                rollout = self._replay_pool.get_recent_paths()
                assert(len(rollout) == 1)
                rollout = rollout[0]
                rollouts_i.append(rollout)

            lengths = [len(r['dones']) for r in rollouts_i]
            print('length: min {0:.1f}, mean {1:.1f}, max {2:.1f}'.format(np.min(lengths), np.mean(lengths), np.max(lengths)))
            rollouts.append(rollouts_i)

        mypickle.dump({'rollouts': rollouts}, self._save_file)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, choices=('SquareClutteredEnv', 'SquareClutteredHoldoutEnv',))
    parser.add_argument('steps', type=int)
    parser.add_argument('savefile', type=str)
    parser.add_argument('-H', type=int)
    parser.add_argument('-K', type=int)
    args = parser.parse_args()

    logger.setup_logger('/tmp/log.txt', 'debug')

    if args.env == 'SquareClutteredEnv':
        env = create_env("SquareClutteredEnv(params={'hfov': 120, 'do_back_up': True, 'collision_reward_only': True, 'collision_reward': -1, 'speed_limits': [2., 2.]})")
    elif args.env == 'SquareClutteredHoldoutEnv':
        env = create_env("SquareClutteredHoldoutEnv(params={'hfov': 120, 'do_back_up': True, 'collision_reward_only': True, 'collision_reward': -1, 'speed_limits': [2., 2.]})")
    else:
        raise NotImplementedError

    gmpcd = GatherMPCData(env, args.steps, args.savefile, args.H, args.K)
    gmpcd.run()

