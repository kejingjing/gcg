import random
import numpy as np
import tensorflow as tf

try:
    from gcg.envs.sim_rccar.square_env import SquareEnv
    from gcg.envs.sim_rccar.square_cluttered_env import SquareClutteredEnv
    from gcg.envs.sim_rccar.square_cluttered_colored_env import SquareClutteredColoredEnv
    from gcg.envs.sim_rccar.square_cluttered_extra_env import SquareClutteredExtraEnv
    from gcg.envs.sim_rccar.cylinder_env import CylinderEnv
    from gcg.envs.sim_rccar.room_cluttered_env import RoomClutteredEnv
    from gcg.envs.sim_rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv
    from gcg.envs.sim_rccar.simple_room_backup_env import SimpleRoomBackupEnv
    from gcg.envs.sim_rccar.forest_env import ForestEnv
except:
    print('Not importing sim_rccar')

try:
    from gcg.envs.rw_rccar.rw_rccar_env import RWrccarEnv
except:
    print('Not importing rw_rccar')

def create_env(env_dict, seed=None):
    if seed is not None:
        print("Creating a {0} environment with random seed {1}".format(env_dict['class'], seed))
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    EnvClass = eval(env_dict['class'])
    env = EnvClass(params=env_dict['params'])

    return env
