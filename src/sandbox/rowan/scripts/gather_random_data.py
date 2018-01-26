import os, argparse

from gcg.data.logger import logger
from gcg.envs.env_utils import create_env
from gcg.sampler.sampler import Sampler
from gcg.data import mypickle

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

    def _itr_save_file(self, itr):
        path, ext = os.path.splitext(self._save_file)
        return '{0}_{1:02d}{2}'.format(path, itr, ext)

    def run(self):
        rollouts = []
        itr = 0

        self._sampler.reset()
        step = 0
        while step < self._steps:
            self._sampler.step(step, take_random_actions=True)
            step += 1

            rollouts += self._sampler.get_recent_paths()
            if step > 0 and step % 5000 == 0 and len(rollouts) > 0:
                mypickle.dump({'rollouts': rollouts}, self._itr_save_file(itr))
                rollouts = []
                itr += 1

        if len(rollouts) > 0:
            mypickle.dump({'rollouts': rollouts}, self._itr_save_file(itr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, choices=('SquareClutteredEnv',
                                                  'SquareClutteredColoredEnv',
                                                  'SquareClutteredConeEnv'))
    parser.add_argument('steps', type=int)
    args = parser.parse_args()

    logger.setup(display_name='gather_random_data', log_path='/tmp/log.txt', lvl='debug')

    if args.env == 'SquareClutteredEnv':
        env = create_env("SquareClutteredEnv(params={'hfov': 120, 'do_back_up': True, 'collision_reward_only': True, 'collision_reward': -1, 'speed_limits': [2., 2.]})")
    elif args.env == 'SquareClutteredColoredEnv':
        env = create_env("SquareClutteredColoredEnv(params={'hfov': 120, 'do_back_up': True, 'collision_reward_only': True, 'collision_reward': -1, 'speed_limits': [2., 2.]})")
    elif args.env == 'SquareClutteredConeEnv':
        env = create_env("SquareClutteredConeEnv(params={'hfov': 120, 'do_back_up': False, 'collision_reward_only': True, 'collision_reward': -1, 'speed_limits': [2., 2.]})")
    else:
        raise NotImplementedError

    curr_dir = os.path.realpath(os.path.dirname(__file__))
    data_dir = os.path.join(curr_dir[:curr_dir.find('gcg/src')], 'gcg/data')
    assert (os.path.exists(data_dir))
    fname = '{0}_random{1:d}.pkl'.format(args.env, args.steps)
    save_dir = os.path.join(data_dir, 'bnn/datasets', os.path.splitext(fname)[0])
    os.makedirs(save_dir, exist_ok=True)
    savefile = os.path.join(save_dir, fname)

    grd = GatherRandomData(env, args.steps, savefile)
    grd.run()

