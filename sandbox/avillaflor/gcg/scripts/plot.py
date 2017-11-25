import joblib
import yaml
import os
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

class Plot:

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
        self._rollouts = self._get_rollouts() 
        self._rollouts_eval = self._get_rollouts(testing=True) 

    #############
    ### Files ###
    #############

    @property
    def _image_folder(self):
        path = os.path.join(self._save_dir, 'analysis_images')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _plot_stats_file(self, testing=False):
        if testing:
            return os.path.join(self._image_folder, '{0}_{1}'.format(self._name, 'stats_testing.png'))
        else:
            return os.path.join(self._image_folder, '{0}_{1}'.format(self._name, 'stats.png'))

    def _plot_trajectories_file(self, itr, testing=False):
        if testing:
            suffix = 'testing'
        else:
            suffix = ''
        return os.path.join(self._image_folder, 'trajectories_{0}_{1}.png'.format(itr, suffix))

    ###############
    ### Getters ###
    ###############

    def get_name(self):
        return self._name

    def get_rollouts(self):
        return self._rollouts

    def get_rollouts_eval(self):
        return self._rollouts_eval
    
    #######################
    ### Data Processing ###
    #######################

    def _get_itr_rollouts(self, itr, testing=False):
        if testing:
            fname = "itr_{0}_rollouts_eval.pkl".format(itr)
        else:
            fname = "itr_{0}_rollouts.pkl".format(itr)
        path = os.path.join(self._data_dir, fname)
        if os.path.exists(path):
            rollouts_dict = joblib.load(path)
            rollouts = rollouts_dict['rollouts']
            return rollouts
        else:
            return None

    def _get_rollouts(self, testing=False):
        itr = 0
        rollouts = []
        itr_rollouts = []
        while itr_rollouts is not None:
            itr_rollouts = self._get_itr_rollouts(itr, testing=testing)
            if itr_rollouts is not None:
                rollouts.append(itr_rollouts)
                itr += 1
        return rollouts

    def get_stats_data(self, testing=False):
        if testing:
            rollouts = self._rollouts_eval
        else:
            rollouts = self._rollouts

        crashes = []
        avg_crashes = []
        rewards = []
        avg_rewards = []
        times = []
        time_tot = 0
        for itr, rollout in enumerate(rollouts):
            for trajectory in rollout:
                env_infos = trajectory['env_infos']
                reward = sum([env_info['reward'] for env_info in env_infos])
                coll = int(env_infos[-1]['coll'])
                rewards.append(reward)
                crashes.append(coll)
                for time in range(len(trajectory['rewards'])):
                    time_tot += 1./240.
                    times.append(time_tot)
                    avg_crashes.append(np.mean(crashes[-120:]))
                    avg_rewards.append(np.mean(rewards[-120:]))

        return avg_crashes, avg_rewards, times

    ################
    ### Plotting ###
    ################

    def plot_stats(self, testing=False):
        avg_crashes, avg_rewards, times = self.get_stats_data(testing=testing)
        f, axes = plt.subplots(2, 1, sharex=True, figsize=(15,15))
        axes[0].set_title('Percent Crashes')
        axes[0].set_ylabel('% Crashes')
        axes[1].set_title('Reward over time')
        axes[1].set_ylabel('R')
        axes[1].set_xlabel('min')
        axes[0].plot(times, avg_crashes)
        axes[1].plot(times, avg_rewards)
        f.savefig(self._plot_stats_file(testing)) 
        plt.close()
#
    def plot_trajectories(self, testing=False):
        blue_line = matplotlib.lines.Line2D([], [], color='b', label='collision')
        red_line = matplotlib.lines.Line2D([], [], color='r', label='no collision')
        if testing:
            rollouts = self._rollouts_eval
        else:
            rollouts = self._rollouts
        for itr, rollout in enumerate(rollouts):
            plt.figure()
            if testing:
                plt.title('Trajectories for testing itr {0}'.format(itr))
            else:
                plt.title('Trajectories for itr {0}'.format(itr))
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
            plt.savefig(self._plot_trajectories_file(itr, testing)) 
            plt.close()

############
### Main ###
############

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--plot_traj', action='store_true')
    args = parser.parse_args()
    plot = Plot(args.data_dir, save_dir=args.save_dir)
    if args.plot_traj:
        plot.plot_trajectories()
        plot.plot_trajectories(testing=True)
    plot.plot_stats()
    plot.plot_stats(testing=True)
