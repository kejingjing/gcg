### our imports
from collections import OrderedDict
import copy

import numpy as np

from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.envs.gibson import transformations as tft

### gibson imports
import pybullet
from transforms3d import quaternions

from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv
# from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Husky
from gibson.core.render.profiler import Profiler


class HuskyEnv(CameraRobotEnv):

    def __init__(self, params={}):
        params.setdefault('mode', 'headless')
        params.setdefault('model_id', '17DRP5sb8fy')
        params.setdefault('timestep', 0.01)
        params.setdefault('frame_skip', 25)
        params.setdefault('resolution', 128)
        params.setdefault('fov', 1.57)
        params.setdefault('gpu', 0)
        params.setdefault('use_filler', True)
        params.setdefault('output', ['nonviz_sensor', 'rgb_filled', 'semantics'])
        params.setdefault('semantic_source', 2)
        params.setdefault('semantic_color', 1)
        params.setdefault('display_ui', False)
        params.setdefault('ui_components', [])
        params.setdefault('random', {'random_initial_pose' : False,
                                     'random_target_pose' : False,
                                     'random_init_x_range': [-0.1, 0.1],
                                     'random_init_y_range': [-0.1, 0.1],
                                     'random_init_z_range': [-0.1, 0.1],
                                     'random_init_rot_range': [-0.1, 0.1],
                                     'random_target_range': 0.1})

        for key, value in params.items():
            setattr(self, key, value)

        self.config = params
        self.gui = False
        self.tracking_camera = {
            'yaw': 20,
            'z_offset': 0.5,
            'distance': 1,
            'pitch': -20
        }

        CameraRobotEnv.__init__(
            self,
            config=params,
            gpu_count=params['gpu'],
            scene_type='building',
            use_filler=params['use_filler'])

        self.robot_introduce(
            Husky(
                is_discrete=False,
                initial_pos=[0, 0, 1.37],
                initial_orn=[0.3, 4.5, 1.2],
                resolution=params['resolution'],
                env=self)
        )

        self.scene_introduce()

        self._setup_spec()

    def _setup_spec(self):
        self.action_spec = OrderedDict()
        self.action_selection_spec = OrderedDict()
        self.observation_vec_spec = OrderedDict()
        self.goal_spec = OrderedDict()

        lows, highs = [], []
        for i, link in enumerate(self.robot.foot_list):
            lows.append(self.robot.action_space.low[i])
            highs.append(self.robot.action_space.high[i])
            self.action_spec[link] = Box(low=lows[-1], high=highs[-1])
        self.action_space = Box(low=np.array(lows), high=np.array(highs))

        self.action_selection_spec = copy.deepcopy(self.action_spec)
        self.action_selection_space = copy.deepcopy(self.action_space)

        assert (np.logical_and(self.action_selection_space.low >= self.action_space.low,
                               self.action_selection_space.high <= self.action_space.high).all())

        self.observation_im_space = Box(low=0, high=255, shape=[self.resolution, self.resolution, 3])
        self.observation_vec_spec['coll'] = Discrete(1)
        self.observation_vec_spec['heading'] = Box(low=0, high=2 * 3.14)
        self.observation_vec_spec['speed'] = Box(low=-0.4, high=0.4)

    def _get_observation_im(self):
        ## Select the nearest points
        eye_pos = self.robot.eyes.current_position()
        x, y, z ,w = self.robot.eyes.current_orientation()
        eye_quat = quaternions.qmult([w, x, y, z], self.robot.eye_offset_orn)
        pose = [eye_pos, eye_quat]

        all_dist, all_pos = self.r_camera_rgb.rankPosesByDistance(pose)
        top_k = self.find_best_k_views(pose[0], all_dist, all_pos)

        render_rgb, render_depth, render_semantics, render_normal, render_unfilled = \
            self.r_camera_rgb.renderOffScreen(pose, top_k)

        if self.config["display_ui"]:
            with Profiler("Rendering visuals: render to visuals"):
                self.render_to_UI()
                self.save_frame += 1
        elif self.gui:
            # Speed bottleneck 2, 116fps
            self.r_camera_rgb.renderToScreen()

        return render_rgb, render_depth, render_semantics

    def _get_observation_vec(self):
        coll = len([pt for pt in self.robot.parts['base_link'].contact_list() if pt[6][2] > 0.15]) > 0
        heading = tft.euler_from_matrix(quaternions.quat2mat(self.robot.get_orientation()))
        speed = np.linalg.norm(env.robot.parts['base_link'].speed())

        return np.array([coll, heading, speed])

    def _get_reward(self):
        pass

    def _get_done(self):
        pass

    def _step(self, a):

        self.robot.apply_action(a)
        self.scene.global_step()
        sensor_state = self.robot.calc_state()


        robot_pos = self.robot.get_position()
        pybullet.resetBasePositionAndOrientation(self.robot_mapId, [robot_pos[0] / self.robot.mjcf_scaling,
                                                                    robot_pos[1] / self.robot.mjcf_scaling, 6],
                                                 [0, 0, 0, 1])

        return None # TODO

    # def step(self, action):
    #     pass
    #
    # def reset(self):
    #     pass

if __name__ == '__main__':
    env = HuskyEnv()
    import IPython; IPython.embed()

"""
How to set pose?
- env.robot.robot_body.reset_pose(pose)
- env.robot.robot_body.reset_position(position)
- env.robot.robot_body.reset_orientation(orientation)

How to get pose?
- env.robot.get_position()
- env.robot.get_orientation() [x, y, z, w]

How to get RGB image?
-

How to get joints?
- env.robot.ordered_joints
"""
