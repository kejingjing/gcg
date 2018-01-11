import os
import joblib
import argparse
import yaml

from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab import config

from avillaflor.gcg.envs.env_utils import create_env
from avillaflor.gcg.policies.mac_policy import MACPolicy
from avillaflor.gcg.policies.rccar_sensors_mac_policy import RCcarSensorsMACPolicy
from avillaflor.gcg.sampler.sampler import RNNCriticSampler

class EvalExp(object):
    def __init__(self, folder, num_rollouts, yaml_file):
        self._folder = folder
        self._num_rollouts = num_rollouts

        ### load data
        with open(yaml_file, 'r') as f:
            params = yaml.load(f)
        
        self._name = params['exp_name']
        self._save_yaml(params)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU
        config.USE_TF = True

        normalize_env = params['alg'].pop('normalize_env')

        env_str = params['alg'].pop('env')
        env = create_env(env_str, is_normalize=normalize_env, seed=params['seed'])

        if 'max_path_length' in params['alg']:
            max_path_length = params['alg'].pop('max_path_length')
        else:
            max_path_length = env.horizon

        policy_class = params['policy']['class']
        PolicyClass = eval(policy_class)
        policy_params = params['policy'][policy_class]

        self._policy = PolicyClass(
            env_spec=env.spec,
            exploration_strategies=params['alg'].pop('exploration_strategies'),
            **policy_params,
            **params['policy']
        )
        
        self._save_rollouts = params['alg']['save_rollouts']
        self._save_rollouts_observations = params['alg']['save_rollouts_observations']

        self._sampler = RNNCriticSampler(
            policy=self._policy,
            env=env,
            n_envs=1, # TODO maybe multiple envs
#            n_envs=params['alg']['n_envs'],
            replay_pool_size=params['alg']['replay_pool_size'],
            max_path_length=max_path_length,
            sampling_method=params['alg']['replay_pool_sampling'],
            save_rollouts=params['alg']['save_rollouts'],
            save_rollouts_observations=params['alg']['save_rollouts_observations'],
            save_env_infos=params['alg']['save_env_infos'],
            env_str=env_str,
            replay_pool_params=params['alg']['replay_pool_params']
        )

    #############
    ### Files ###
    #############

    def _itr_file(self, itr):
        return os.path.join(self._folder, 'itr_{0:d}.pkl'.format(itr))

    #########################
    ### Save/Load methods ###
    #########################

    def _save_eval_rollouts(self, itr, rollouts):
        folder_name = os.path.join(self._folder, self._name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        fname = os.path.join(folder_name, 'itr_{0:d}_exp_eval.pkl'.format(itr))
        joblib.dump({'rollouts': rollouts}, fname, compress=3)

    def _save_yaml(self, params):
        folder_name = os.path.join(self._folder, self._name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        fname = os.path.join(folder_name, 'params.yaml')
        with open(fname, 'w') as f:
            yaml.dump(params, f)


    def _load_model_file(self, itr):
        fname = 'itr_{0}.ckpt'.format(itr)
        fname = os.path.join(self._folder, fname)
        self._policy.restore_ckpt(fname)
    
    ########################
    ### Training methods ###
    ########################

    def eval_policy(self, itr=-1):
        if itr == -1:
            itr = 0
            while os.path.exists(self._itr_file(itr)):
                itr += 1
            itr -= 1

        self._load_model_file(itr)
        
        logger.log('Evaluating policy for itr {0}'.format(itr))
        rollouts = []
        step = 0
        n_envs = 1 # TODO maybe multiple
        logger.log('Starting rollout {0}'.format(len(rollouts)))
        while len(rollouts) < self._num_rollouts:
            self._sampler.step(step)
            step += n_envs
            new_rollouts = self._sampler.get_recent_paths()
            if len(new_rollouts) > 0:
                rollouts += new_rollouts
                logger.log('Starting rollout {0}'.format(len(rollouts)))
                self._save_eval_rollouts(itr, rollouts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('numrollouts', type=int)
    parser.add_argument('--yaml', type=str, default='yamls/ours.yaml')
    args = parser.parse_args()
    
    eval_exp = EvalExp(args.folder, args.numrollouts, args.yaml)
    eval_exp.eval_policy()
