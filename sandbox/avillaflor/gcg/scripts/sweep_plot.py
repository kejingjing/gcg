import joblib
import yaml
import os
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sandbox.avillaflor.gcg.scripts.plot import Plot

class MultiPlot:

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
        self._plots = []
        self._names = []
        for f in os.listdir(self._data_dir):
            plot = Plot(os.path.join(self._data_dir, f))
            if len(plot.get_rollouts()) > 0:
                self._plots.append(plot)
                self._names.append(plot.get_name())

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

    ################
    ### Plotting ###
    ################

    def plot_stats(self, testing=False):
        f, axes = plt.subplots(2, 1, sharex=True, figsize=(15,15))
        lines = []
        axes[0].set_title('Crashes per time')
        axes[0].set_ylabel('Crashes')
        axes[1].set_title('Reward over time')
        axes[1].set_ylabel('R')
        axes[1].set_xlabel('min')
        
        for plot in self._plots:
            if testing:
                rollouts = plot.get_rollouts_eval()
            else:
                rollouts = plot.get_rollouts()
            
            crashes = []
            avg_crashes = []
            rewards = []
            avg_rewards = []
            times = []
            time_tot = 0
            for itr, rollout in enumerate(rollouts):
                num_coll = 0.
                itr_reward = 0.
                itr_time = 0
                for trajectory in rollout:
                    env_infos = trajectory['env_infos']
                    rs = [env_info['reward'] for env_info in env_infos]
                    num_coll += int(env_infos[-1]['coll'])
                    itr_reward += sum(rs)
                    itr_time += len(rs) / 240.
                time_tot += itr_time
                crashes.append(num_coll/len(rollout))
                avg_crashes.append(np.mean(crashes[-12:]))
                times.append(time_tot)
                rewards.append(itr_reward/len(rollout))
                avg_rewards.append(np.mean(rewards[-12:]))

            line, = axes[0].plot(times, avg_crashes)
            lines.append(line)
            axes[1].plot(times, avg_rewards)
        
        f.legend(lines, self._names)
        f.savefig(self._plot_stats_file(testing)) 
        plt.close()
    
############
### Main ###
############

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    plot = MultiPlot(args.data_dir, save_dir=args.save_dir)
    plot.plot_stats()
    plot.plot_stats(testing=True)
