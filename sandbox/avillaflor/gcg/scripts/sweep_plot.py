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
            path1 = os.path.join(self._data_dir, f)
            seed_plots = []
            for f2 in os.listdir(path1):
                path2 = os.path.join(path1, f2)
                plot = Plot(path2)
                if len(plot.get_rollouts()) > 0:
                    seed_plots.append(plot)
            if len(seed_plots) > 0:
                self._plots.append(seed_plots)
                self._names.append(f)

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
       
        datas_list = []

        for seed_plots in self._plots:
            seed_datas = []
            for plot in seed_plots:
                data = plot.get_stats_data(testing=testing)
                seed_datas.append(data)
            datas_list.append(seed_datas)

        for seed_datas in datas_list:
            avg_crashes = []
            avg_rewards = []
            times = []
            for time in range(min([len(data[2]) for data in seed_datas])):
                times.append(time / 240.)
                avg_crashes.append(np.mean([data[0][time] for data in seed_datas]))
                avg_rewards.append(np.mean([data[1][time] for data in seed_datas]))

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
