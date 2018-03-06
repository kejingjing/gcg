import os, time
from collections import defaultdict

import numpy as np

import rosbag

from gcg.algos.async_gcg import AsyncGCG, run_async_gcg_train, run_async_gcg_inference
from gcg.data.logger import logger
from gcg.envs.vec_env_executor import VecEnvExecutor

class AsyncCrazyflieGCG(AsyncGCG):
    def __init__(self, **kwargs):
        AsyncGCG.__init__(self, **kwargs)

        assert (self._eval_sampler is None)

    ########################
    ### Training methods ###
    ########################

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
            # st = time.time()
            self._sampler.step(inference_step,
                               take_random_actions=(inference_step <= self._onpolicy_after_n_steps),
                               explore=True)
            inference_step += self._sampler.n_envs

            # elapsed_t = time.time() - st
            # print("Elapsed in inference:", elapsed_t)
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


def run_async_crazyflie_gcg_train(params, is_continue):
    run_async_gcg_train(params, is_continue, AsyncClass=AsyncCrazyflieGCG)


def run_async_crazyflie_gcg_inference(params, is_continue):
    run_async_gcg_inference(params, is_continue, AsyncClass=AsyncCrazyflieGCG)
