try:
    from gcg.envs.sim_rccar.square_env import SquareEnv
    from gcg.envs.sim_rccar.square_cluttered_env import SquareClutteredEnv
    from gcg.envs.sim_rccar.square_cluttered_holdout_env import SquareClutteredHoldoutEnv
    from gcg.envs.sim_rccar.cylinder_env import CylinderEnv
    from gcg.envs.sim_rccar.room_cluttered_env import RoomClutteredEnv
    from gcg.envs.sim_rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv
    from gcg.envs.sim_rccar.simple_room_backup_env import SimpleRoomBackupEnv
except:
    print('Not importing sim_rccar')

try:
    from gcg.envs.rw_rccar.rw_rccar_env import RWrccarEnv
except:
    print('Not importing rw_rccar')

def create_env(env_str, seed=None):
    env = eval(env_str)

    return env
