from .rccar.square_env import SquareEnv
from .rccar.square_cluttered_env import SquareClutteredEnv
from .rccar.cylinder_env import CylinderEnv
from .rccar.room_cluttered_env import RoomClutteredEnv
from .rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv

def create_env(env_str, seed=None):
    env = eval(env_str)

    return env
