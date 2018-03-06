import yaml
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gcg.envs.sim_rccar.forest_env import ForestEnv
from gcg.data import mypickle

class HighRes:
    def __init__(self, data_path, save_dir, env):
        self._data_path = data_path
        self._save_dir = save_dir
        self._env = env
        self._rollouts, self._goals = self._get_rollouts()

    #############
    ### Files ###
    #############

    def _itr_folder(self, itr):
        path = os.path.join(self._save_dir, 'itr_{0}'.format(itr))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _plot_traj_file(self, itr):
        folder = self._itr_folder(itr)
        return os.path.join(folder, 'traj.png')

    def _high_res_image_file(self, itr, step):
        folder = self._itr_folder(itr)
        return os.path.join(folder, 'image_{0}.png'.format(step))
    
    ############
    ### Data ###
    ############

    def _get_rollouts(self):
        data = mypickle.load(self._data_path)
        rollouts = []
        goals = []
        for traj in data['rollouts']:
            pos_hprs = []
            traj_goals = []
            for step in traj['env_infos'][:-1]:
                pos_hprs.append((step['pos'], step['hpr']))
                traj_goals.append(step['goal_h'])
            rollouts.append(pos_hprs)
            goals.append(traj_goals)
        return rollouts, goals
    
    ################
    ### Plotting ###
    ################

    def plot_trajectories(self):
        for itr, rollout in enumerate(self._rollouts):
            plt.figure()
            plt.title('Trajectory {0}'.format(itr))
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.ylim([-12.5, 12.5])
            plt.xlim([-12.5, 12.5])
            pos_x = []
            pos_y = []
            for step, goal in zip(rollout, self._goals[itr]):
                pos_x.append(step[0][0])
                pos_y.append(step[0][1])
            ax = plt.axes()
            goal_x = pos_x[0] - 22.5 * np.sin(goal)
            goal_y = pos_y[0] + 22.5 * np.cos(goal)
#            ax.arrow(pos_x[0], pos_y[0], goal_x - pos_x[0], goal_y - pos_y[0], head_width=1.25, head_length=2.5)
            ax.arrow(pos_x[0], pos_y[0], -4 * np.sin(goal), 4 * np.cos(goal), head_width=0.25, head_length=1.0, fc='g', ec='g')
            plt.scatter(pos_x, pos_y, s=1)
#            plt.scatter(goal_x, goal_y, s=100, marker='x')
            plt.savefig(self._plot_traj_file(itr))

    def take_hr_images(self):
        for itr, rollout in enumerate(self._rollouts):
            for i, step in enumerate(rollout):
                pos = step[0]
                hpr = step[1]
                goal = self._goals[itr][i]
                self._env._place_vehicle(pos, hpr)
                heading = self._env._get_heading()
                angle = heading - goal
                image = self._env.get_front_cam_obs()
                plt.figure()
                plt.imshow(image)
                ax = plt.axes()
                ax.arrow(640, 480, 120 * np.sin(angle), -120 * np.cos(angle), head_width=50., head_length=25., width=25., fc='k', ec='k')
                plt.savefig(self._high_res_image_file(itr, i)) 

############
### Main ###
############

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()
    params = {'size': [1280, 720]}
    env = ForestEnv(params=params)
    hr = HighRes(args.data_path, args.save_dir, env)
    hr.plot_trajectories()
    hr.take_hr_images()
