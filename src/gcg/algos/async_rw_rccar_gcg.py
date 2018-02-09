import os, time
from collections import defaultdict

import numpy as np

import rosbag

from gcg.algos.async_gcg import AsyncGCG, run_async_gcg_train, run_async_gcg_inference
from gcg.data.logger import logger
from gcg.envs.vec_env_executor import VecEnvExecutor

class AsyncRWrccarGCG(AsyncGCG):
    def __init__(self, **kwargs):
        self._added_rosbag_filenames = []

        AsyncGCG.__init__(self, **kwargs)

        assert (self._eval_sampler is None)

    #############
    ### Files ###
    #############

    @property
    def _rosbag_dir(self):
        return os.path.join(self._save_dir, 'rosbags')

    def _rosbag_file_name(self, num):
        return os.path.join(self._rosbag_dir, 'rosbag{0:04d}.bag'.format(num))

    ###############
    ### Restore ###
    ###############

    def _add_rosbags(self, rosbag_filenames):
        """ Convert rosbags to rollouts and add to sampler """
        def visualize(rollout):
            r_len = len(rollout['dones'])
            import matplotlib.pyplot as plt

            f, ax = plt.subplots(1, 1)

            imshow = ax.imshow(rollout['observations_im'][0][:, :, 0], cmap='Greys_r')
            ax.set_title('r: {0}'.format(rollout['rewards'][0]))
            plt.show(block=False)
            plt.pause(0.01)
            input('t: 0')
            for t in range(1, r_len):
                imshow.set_data(rollout['observations_im'][t][:, :, 0])
                ax.set_title('r: {0}'.format(rollout['rewards'][t]))
                f.canvas.draw()
                plt.pause(0.01)
                input('t: {0}'.format(t))

            plt.close(f)

        timesteps_kept = 0
        timesteps_total = 0
        for fname in rosbag_filenames:
            self._added_rosbag_filenames.append(fname)

            ### read bag file
            try:
                bag = rosbag.Bag(fname, 'r')
            except:
                logger.warn('{0}: could not open'.format(os.path.basename(fname)))
                continue
            d_bag = defaultdict(list)
            for topic, msg, t in bag.read_messages():
               d_bag[topic].append(msg)
            bag.close()

            timesteps_total += len(d_bag['mode']) - 1

            ### trim to whenever collision occurs
            colls = np.array([msg.data for msg in d_bag['collision/all']])
            if len(colls) == 0:
                logger.warn('{0}: empty bag'.format(os.path.basename(fname)))
                continue
            if colls.max() > 0:
                if colls.sum() > 1:
                    logger.warn('{0}: has multiple collisions'.format(os.path.basename(fname)))
                    continue
                if colls[-1] != 1:
                    logger.warn('{0}: has collision, but does not end in collision'.format(os.path.basename(fname)))
                    continue

            ### make sure it moved at least a little bit
            encoders = np.array([msg.data for msg in d_bag['encoder/both']])
            if (abs(encoders) > 1e-4).sum() < 2:
                logger.warn('{0}: car never moved'.format(os.path.basename(fname)))
                continue

            ### update env and step
            def update_env(t):
                for key in d_bag.keys():
                    try:
                        self._env.ros_msg_update(d_bag[key][t], [key])
                    except:
                        import IPython; IPython.embed()

            update_env(0)
            if len(self._sampler) == 0:
                logger.warn('Resetting!')
                self._sampler.reset(offline=True)

            bag_length = len(d_bag['mode'])
            for t in range(1, bag_length):
                update_env(t)
                action = np.array([d_bag['cmd/steer'][t-1].data, d_bag['cmd/vel'][t-1].data])
                self._sampler.step(len(self._sampler), actions=[action], offline=True)

            if not self._sampler.is_done_nexts:
                logger.warn('{0}: did not end in done, manually resetting'.format(os.path.basename(fname)))
                self._sampler.reset(offline=True)

            # if not self._env._is_collision:
            #     logger.warn('{0}: not ending in collision'.format(os.path.basename(fname)))

            timesteps_kept += len(d_bag['mode']) - 1

        logger.info('Adding {0:d} timesteps ({1:.2f} kept)'.format(timesteps_kept, timesteps_kept / float(timesteps_total)))

    def _add_offpolicy(self, folders, max_to_add):
        for folder in folders:
            assert (os.path.exists(folder))
            logger.info('Loading rosbag data from {0}'.format(folder))
            rosbag_filenames = sorted([os.path.join(folder, fname) for fname in os.listdir(folder) if '.bag' in fname])
            self._add_rosbags(rosbag_filenames)
        logger.info('Added {0} samples'.format(len(self._sampler)))

    def _restore_train_rollouts(self):
        """
        :return: iteration that it is currently on
        """
        rosbag_num = 0
        rosbag_filenames = []
        while True:
            fname = self._rosbag_file_name(rosbag_num)
            if not os.path.exists(fname):
                break

            rosbag_num += 1
            if fname in self._added_rosbag_filenames:
                continue # don't add already added rosbag filenames

            rosbag_filenames.append(fname)

        if len(rosbag_filenames) > 0:
            logger.info('Restoring {0} rosbags....'.format(rosbag_num))
            self._add_rosbags(rosbag_filenames)
            logger.info('Done restoring rosbags!')

    #####################
    ### Async methods ###
    #####################

    @property
    def _rsync_send_includes(self):
        return super(AsyncRWrccarGCG, self)._rsync_send_includes + ['rosbags/'] + ['rosbags/*.bag']

    ########################
    ### Training methods ###
    ########################

    def _train_load_data(self, inference_itr):
        new_inference_itr = self._get_inference_itr()
        if inference_itr < new_inference_itr:
            for i in range(inference_itr, new_inference_itr):
                try:
                    logger.debug('Loading files for itrs [{0}, {1}]'.format(inference_itr + 1, new_inference_itr))
                    self._restore_train_rollouts()
                    inference_itr = new_inference_itr
                except:
                    logger.debug('Failed to load files for itr {0}'.format(i))

        return inference_itr

    def _inference_reset_sampler(self, keep_rosbag=True):
        while True:
            try:
                self._sampler.reset(keep_rosbag=keep_rosbag)
                break
            except Exception as e:
                logger.warn('Reset exception {0}'.format(str(e)))
                while not self._env.ros_is_good(print=False):  # TODO hard coded
                    time.sleep(0.25)
                logger.warn('Continuing...')

    def _inference_step(self, inference_step):
        try:
            self._sampler.step(inference_step,
                               take_random_actions=(inference_step <= self._onpolicy_after_n_steps),
                               explore=True)
            inference_step += self._sampler.n_envs
        except Exception as e:
            logger.warn('Sampler exception {0}'.format(str(e)))
            trashed_steps = self._sampler.trash_current_rollouts()
            inference_step -= trashed_steps
            logger.warn('Trashed {0} steps'.format(trashed_steps))
            while not self._env.ros_is_good(print=False):
                time.sleep(0.25)
            self._inference_reset_sampler(keep_rosbag=False)
            logger.warn('Continuing...')

        return inference_step


def run_async_rw_rccar_gcg_train(params, is_continue):
    run_async_gcg_train(params, is_continue, AsyncClass=AsyncRWrccarGCG)


def run_async_rw_rccar_gcg_inference(params, is_continue):
    run_async_gcg_inference(params, is_continue, AsyncClass=AsyncRWrccarGCG)
