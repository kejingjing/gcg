from gcg.envs.rccar.square_env import SquareEnv
from gcg.envs.rccar.square_cluttered_env import SquareClutteredEnv
from gcg.envs.rccar.cylinder_env import CylinderEnv
from gcg.envs.rccar.room_cluttered_env import RoomClutteredEnv
from gcg.envs.rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv
from gcg.envs.rccar.simple_room_backup_env import SimpleRoomBackupEnv
from gcg.envs.rccar.forest_env import ForestEnv

def create_env(env_str, seed=None):
    env = eval(env_str)

    return env
