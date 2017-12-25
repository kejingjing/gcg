import joblib
import yaml
import os
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sandbox.avillaflor.gcg.scripts.plot import Plot

class EvalExpPlot(Plot):
    # TODO STATS DATA

    def __init__(self, data_dir, save_dir=None):
        self._data_dir = data_dir
        if save_dir is None:
            self._save_dir = data_dir
        else:
            self._save_dir = save_dir
        if self._data_dir[-1] == '/':
            self._name = os.path.split(self._data_dir[:-1])[-1]
        else:
            self._name = os.path.split(self._data_dir)[-1]
        
        self._rollouts, self._names = self._get_rollouts() 

    #############
    ### Files ###
    #############

    def _plot_trajectories_file(self, name):
        return os.path.join(self._image_folder, 'trajectories_{0}.png'.format(name))

    ###############
    ### Getters ###
    ###############

    def get_name(self):
        return self._name

    def get_names(self):
        return self._names

    def get_rollouts(self):
        return self._rollouts

    #######################
    ### Data Processing ###
    #######################

    def _get_rollouts(self):
        rollouts = []
        names = []
        for fname1 in os.listdir(self._data_dir):
            path1 = os.path.join(self._data_dir, fname1)
            if os.path.isdir(path1):
                for fname2 in os.listdir(path1):
                    if os.path.splitext(fname2)[-1] == '.pkl':
                        path2 = os.path.join(path1, fname2)
                        rollouts_dict = joblib.load(path2)
                        rollouts.append(rollouts_dict['rollouts'])
                        names.append(fname1)
                        break
        return rollouts, names

    ################
    ### Plotting ###
    ################

    def plot_trajectories(self):
        blue_line = matplotlib.lines.Line2D([], [], color='b', label='collision')
        red_line = matplotlib.lines.Line2D([], [], color='r', label='no collision')
        for rollout, name in zip(self._rollouts, self._names):
            plt.figure()
            plt.title('Trajectories')
            plt.xlabel('X position')
            plt.ylabel('Y position')
            # TODO
            plt.ylim([-12.5, 12.5])
            plt.xlim([-12.5, 12.5])
            plt.legend(handles=[blue_line, red_line], loc='center')
            for trajectory in rollout:
                env_infos = trajectory['env_infos']
                is_coll = env_infos[-1]['coll']
                pos_x = []
                pos_y = []
                for env_info in env_infos:
                    pos = env_info['pos']
                    pos_x.append(pos[0])
                    pos_y.append(pos[1])
                if is_coll:
                    plt.plot(pos_x, pos_y, color='r')
                    plt.scatter(pos_x[-1], pos_y[-1], marker="x", color='r')
                else:
                    plt.plot(pos_x, pos_y, color='b')
            plt.savefig(self._plot_trajectories_file(name)) 
            plt.close()

############
### Main ###
############

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    plot = EvalExpPlot(args.data_dir, save_dir=args.save_dir)
    plot.plot_trajectories()
