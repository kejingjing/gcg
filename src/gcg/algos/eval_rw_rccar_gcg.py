import os, sys, glob

import numpy as np

from gcg.algos.gcg import GCG
from gcg.envs.env_utils import create_env
from gcg.policies.gcg_policy import GCGPolicy
from gcg.sampler.sampler import Sampler
from gcg.data.logger import logger
from gcg.data import mypickle

from gcg.algos.eval_gcg import EvalGCG, eval_gcg

class EvalRWrccarGCG(EvalGCG):

    def __init__(self, eval_itr, eval_save_dir, **kwargs):
        EvalGCG.__init__(self, eval_itr, eval_save_dir, **kwargs)

    ############
    ### Eval ###
    ############

    def _eval_reset(self, **kwargs):
        while True:
            try:
                self._sampler.reset(**kwargs)
                break
            except Exception as e:
                logger.warn('Reset exception {0}'.format(str(e)))
                logger.info('Press enter to continue')
                input()
                logger.info('')
        
    def _eval_step(self):
        try:
            self._sampler.step(step=0,
                               take_random_actions=False,
                               explore=False)
        except Exception as e:
            logger.warn('Sampler exception {0}'.format(str(e)))
            self._sampler.trash_current_rollouts()

            logger.info('Press enter to continue')
            input()
            self._eval_reset(keep_rosbag=False)

    def _eval_save(self, rollouts, new_rollouts):
        logger.info('')
        logger.info('Keep rollout?')
        response = input()
        if response != 'y':
            logger.info('NOT saving rollouts')
        else:
            logger.info('Saving rollouts')
            rollouts += new_rollouts
            self._save_eval_rollouts(rollouts)

        return rollouts
    
def eval_rw_rccar_gcg(params, itr):
    eval_gcg(params, itr, EvalClass=EvalRWrccarGCG)
