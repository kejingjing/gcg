from avillaflor.gcg.envs.rccar.square_env import SquareEnv
from avillaflor.gcg.envs.rccar.square_cluttered_env import SquareClutteredEnv
from avillaflor.gcg.envs.rccar.cylinder_env import CylinderEnv
from avillaflor.gcg.envs.rccar.room_cluttered_env import RoomClutteredEnv
from avillaflor.gcg.envs.rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv

def create_env(env_str, seed=None):
    env = eval(env_str)

    return env
