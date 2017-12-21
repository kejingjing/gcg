from rllab.misc.ext import set_seed
### environments
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from sandbox.gkahn.gcg.utils import logger

def create_env(env_str, is_normalize=True, seed=None):
    try:
        import gym
        from rllab.envs.gym_env import GymEnv, FixedIntervalVideoSchedule
    except:
        GymEnv = None
        logger.debug('Unable to import gym')

    try:
        from sandbox.gkahn.gcg.envs.rccar.square_env import SquareEnv
        from sandbox.gkahn.gcg.envs.rccar.square_cluttered_env import SquareClutteredEnv
        from sandbox.gkahn.gcg.envs.rccar.cylinder_env import CylinderEnv
    except:
        logger.debug('Unable to import sim rccar')

    try:
        from sandbox.gkahn.gcg.envs.rw_rccar.rw_rccar_env import RWrccarEnv
    except:
       logger.debug('Unable to import rw rccar')

    inner_env = eval(env_str)
    if is_normalize:
        inner_env = normalize(inner_env)
    env = TfEnv(inner_env)

    # set seed
    if seed is not None and GymEnv is not None:
        set_seed(seed)
        if isinstance(inner_env, GymEnv):
            inner_env.env.seed(seed)

    return env
