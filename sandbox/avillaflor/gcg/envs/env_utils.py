from rllab.misc.ext import set_seed
### environments
import gym
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

def create_env(env_str, is_normalize=True, seed=None):
    from rllab.envs.gym_env import GymEnv, FixedIntervalVideoSchedule

    from sandbox.avillaflor.gcg.envs.rccar.square_env import SquareEnv
    from sandbox.avillaflor.gcg.envs.rccar.square_cluttered_env import SquareClutteredEnv
    from sandbox.avillaflor.gcg.envs.rccar.cylinder_env import CylinderEnv
    from sandbox.avillaflor.gcg.envs.rccar.room_cluttered_env import RoomClutteredEnv
    from sandbox.avillaflor.gcg.envs.rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv

    inner_env = eval(env_str)
    if is_normalize:
        inner_env = normalize(inner_env)
    env = TfEnv(inner_env)
    if hasattr(inner_env, 'observation_im_space'):
        env.spec.observation_im_space = inner_env.observation_im_space
    if hasattr(inner_env, 'observation_vec_space'):
        env.spec.observation_vec_space = inner_env.observation_vec_space
    # set seed
    if seed is not None:
        set_seed(seed)
        if isinstance(inner_env, GymEnv):
            inner_env.env.seed(seed)

    return env
