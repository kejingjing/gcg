import gym

#from rllab.misc.ext import set_seed
#from rllab.envs.gym_env import GymEnv
from avillaflor.tf.envs.base import TfEnv
from avillaflor.gcg.envs.rccar.square_env import SquareEnv
from avillaflor.gcg.envs.rccar.square_cluttered_env import SquareClutteredEnv
from avillaflor.gcg.envs.rccar.cylinder_env import CylinderEnv
from avillaflor.gcg.envs.rccar.room_cluttered_env import RoomClutteredEnv
from avillaflor.gcg.envs.rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv

def create_env(env_str, seed=None):
    env = eval(env_str)
    #    inner_env = eval(env_str)
#    env = TfEnv(inner_env)
    # set seed
#    if seed is not None:
#        set_seed(seed)
#        if isinstance(inner_env, GymEnv):
#            inner_env.env.seed(seed)

    return env
