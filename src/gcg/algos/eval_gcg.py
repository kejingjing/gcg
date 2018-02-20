import os, sys

import numpy as np

from gcg.algos.gcg import GCG
from gcg.envs.env_utils import create_env
from gcg.policies.gcg_policy import GCGPolicy
from gcg.sampler.sampler import Sampler
from gcg.data.logger import logger
from gcg.data import mypickle

class EvalGCG(GCG):

    def __init__(self, eval_itr, eval_save_dir, **kwargs):
        self._eval_itr = eval_itr
        self._eval_save_dir = eval_save_dir

        self._save_dir = kwargs['save_dir']
        self._load_dir = kwargs.get('load_dir', self._save_dir)
        self._env_eval = kwargs['env_eval']
        self._policy = kwargs['policy']

        self._sampler = Sampler(
            policy=kwargs['policy'],
            env=kwargs['env_eval'],
            n_envs=1,
            replay_pool_size=int(np.ceil(1.5 * kwargs['max_path_length']) + 1),
            max_path_length=kwargs['max_path_length'],
            sampling_method=kwargs['replay_pool_sampling'],
            save_rollouts=True,
            save_rollouts_observations=True,
            save_env_infos=True,
            env_dict=kwargs['env_dict'],
            replay_pool_params=kwargs['replay_pool_params']
        )

    #############
    ### Files ###
    #############

    def _eval_rollouts_file_name(self, itr):
        return os.path.join(self._eval_save_dir, 'itr_{0:04d}_eval_rollouts.pkl'.format(itr))

    ############
    ### Save ###
    ############

    def _save_eval_rollouts(self, rollouts):
        fname = self._eval_rollouts_file_name(self._eval_save_dir)
        mypickle.dump({'rollouts': rollouts}, fname)

    ###############
    ### Restore ###
    ###############

    def _load_eval_rollouts(self, itr):
        fname = self._eval_rollouts_file_name(itr)
        if os.path.exists(fname):
            rollouts = mypickle.load(fname)['rollouts']
        else:
            rollouts = []
        return rollouts

    ############
    ### Eval ###
    ############

    def eval(self):
        ### Load policy
        policy_fname = self._load_inference_policy_file_name(self._eval_itr)
        if not os.path.exists(policy_fname):
            logger.error('Policy for itr {0} does not exist'.format(self._eval_itr))
            sys.exit(0)
        logger.info('Restoring policy for itr {0}'.format(self._eval_itr))
        self._policy.restore(policy_fname, train=False)

        ### Load previous eval rollouts
        logger.info('Loading previous eval rollouts')
        rollouts = self._load_eval_rollouts(self._eval_itr)
        logger.info('Loaded {0} rollouts'.format(len(rollouts)))

        logger.info('')
        logger.info('Rollout {0}'.format(len(rollouts)))
        while True:
            try:
                self._sampler.step(step=0,
                                   take_random_actions=False,
                                   explore=False)
            except Exception as e:
                logger.warn('Sampler exception {0}'.format(str(e)))
                self._sampler.trash_current_rollouts()

                logger.info('Press enter to continue')
                input()
                logger.info('')
                logger.info('Rollout {0}'.format(len(rollouts)))

            new_rollouts = self._sampler.get_recent_paths()
            if len(new_rollouts) > 0:
                logger.info('')
                logger.info('Keep rollout?')
                response = input()
                if response != 'y':
                    logger.info('NOT saving rollouts')
                    continue

                logger.info('Saving rollouts')
                rollouts += new_rollouts
                self._save_eval_rollouts(rollouts)

                logger.info('')
                logger.info('Rollout {0}'.format(len(rollouts)))

def eval_gcg(params, itr):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir[:curr_dir.find('src/gcg')], 'data')
    assert (os.path.exists(data_dir))
    save_dir = os.path.join(data_dir, params['exp_name'])
    assert (os.path.exists(save_dir))
    eval_save_dir = os.path.join(save_dir, 'eval_itr_{0:04d}'.format(itr))
    os.makedirs(eval_save_dir, exist_ok=True)

    logger.setup(display_name=params['exp_name'],
                 log_path=os.path.join(eval_save_dir, 'log.txt'),
                 lvl=params['log_level'])

    # TODO: set seed

    # copy yaml for posterity
    yaml_path = os.path.join(save_dir, 'params.yaml')
    with open(yaml_path, 'w') as f:
        f.write(params['txt'])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU

    env_eval_dict = params['alg'].pop('env_eval')
    env_eval = create_env(env_eval_dict, seed=params['seed'])

    env_eval.reset()

    #####################
    ### Create policy ###
    #####################

    policy_class = params['policy']['class']
    PolicyClass = eval(policy_class)
    policy_params = params['policy'][policy_class]

    policy = PolicyClass(
        env_spec=env_eval.spec,
        exploration_strategies={},
        **policy_params,
        **params['policy']
    )

    ########################
    ### Create algorithm ###
    ########################

    if 'max_path_length' in params['alg']:
        max_path_length = params['alg'].pop('max_path_length')
    else:
        max_path_length = env_eval.horizon
    algo = EvalGCG(
        save_dir=save_dir,
        env=None,
        env_eval=env_eval,
        policy=policy,
        max_path_length=max_path_length,
        env_dict=env_eval_dict,
        **params['alg']
    )
    algo.eval()
