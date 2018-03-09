try:
    from gcg.envs.sim_rccar.square_env import SquareEnv
    from gcg.envs.sim_rccar.square_cluttered_env import SquareClutteredEnv
    from gcg.envs.sim_rccar.square_cluttered_colored_env import SquareClutteredColoredEnv
    from gcg.envs.sim_rccar.square_cluttered_cone_env import SquareClutteredConeEnv
    from gcg.envs.sim_rccar.cylinder_env import CylinderEnv
    from gcg.envs.sim_rccar.room_cluttered_env import RoomClutteredEnv
    from gcg.envs.sim_rccar.simple_room_cluttered_env import SimpleRoomClutteredEnv
    from gcg.envs.sim_rccar.simple_room_backup_env import SimpleRoomBackupEnv
    from gcg.envs.sim_rccar.forest_env import ForestEnv
    from gcg.envs.sim_rccar.outdoors_env import OutdoorsEnv
except:
    print('Not importing sim_rccar')

try:
    from gcg.envs.rw_rccar.rw_rccar_env import RWrccarEnv
except:
    print('Not importing rw_rccar')

try:
    from gcg.envs.carla.carla_env import CarlaEnv
except:
    print('Not importing carla')

def create_env(env_dict, seed=None):
    EnvClass = eval(env_dict['class'])
    env = EnvClass(params=env_dict['params'])

    return env
