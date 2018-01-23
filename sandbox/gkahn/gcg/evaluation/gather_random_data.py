import argparse

from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.sampler.sampler import Sampler
from sandbox.gkahn.gcg.utils import mypickle

class DummyPolicy(object):
    @property
    def N(self):
        return 1

    @property
    def gamma(self):
        return 1    

    @property
    def obs_history_len(self):
        return 1

    def get_actions(self, **kwargs):
        pass

    def reset_get_action(self):
        pass

class GatherRandomData(object):
    def __init__(self, env, steps, save_file):
        self._env = env
        self._steps = steps
        self._save_file = save_file

        self._sampler = Sampler(
                            policy=DummyPolicy(),
                            env=self._env,
                            n_envs=1,
                            replay_pool_size=steps,
                            max_path_length=self._env.horizon,
                            sampling_method='uniform',
                            save_rollouts=True,
                            save_rollouts_observations=True,
                            save_env_infos=True)

    def run(self):
        self._sampler.reset()
        step = 0
        while step < self._steps:
            self._sampler.step(step, take_random_actions=True)
            step += 1

        rollouts = self._sampler.get_recent_paths()
        mypickle.dump({'rollouts': rollouts}, self._save_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, choices=('SquareClutteredEnv', 'SquareClutteredHoldoutEnv',))
    parser.add_argument('steps', type=int)
    parser.add_argument('savefile', type=str)
    args = parser.parse_args()

    logger.setup_logger('/tmp/log.txt', 'debug')

    if args.env == 'SquareClutteredEnv':
        env = create_env("SquareClutteredEnv(params={'hfov': 120, 'do_back_up': True, 'collision_reward_only': True, 'collision_reward': -1, 'speed_limits': [2., 2.]})")
    elif args.env == 'SquareClutteredHoldoutEnv':
        env = create_env("SquareClutteredHoldoutEnv(params={'hfov': 120, 'do_back_up': True, 'collision_reward_only': True, 'collision_reward': -1, 'speed_limits': [2., 2.]})")
    else:
        raise NotImplementedError

    grd = GatherRandomData(env, args.steps, args.savefile)
    grd.run()

