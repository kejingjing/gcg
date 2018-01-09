import argparse, os, glob
import yaml

import numpy as np
import matplotlib.pyplot as plt

from rllab import config
from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as rllab_logger

from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.sampler.sampler import Sampler
from sandbox.gkahn.gcg.utils import mypickle

class EvalTraversabilityPolicy(object):

    def __init__(self, params):
        self._params = params
        logger.setup_logger(os.path.join(rllab_logger.get_snapshot_dir(), 'log_eval.txt'),
                            params['log_level'])

        self._internal_env = None
        self._internal_policy = None

    @property
    def _env(self):
        if self._internal_env is None:
            logger.info('Creating env')
            self._internal_env = create_env(params['alg']['env'], is_normalize=params['alg']['normalize_env'])

        return self._internal_env

    @property
    def _policy(self):
        if self._internal_policy is None:
            logger.info('Create policy')
            policy = self._create_policy(self._params, self._env)

            logger.info('Restore policy')
            self._restore_policy(policy)

            self._internal_policy = policy

        return self._internal_policy

    def _create_policy(self, params, env):
        from sandbox.gkahn.traversability.policies.traversability_policy import TraversabilityPolicy

        policy_class = params['policy']['class']
        PolicyClass = eval(policy_class)
        policy_params = params['policy'][policy_class]

        policy = PolicyClass(
            env_spec=env.spec,
            inference_only=False,
            **policy_params,
            **params['policy']
        )

        return policy

    def _restore_policy(self, policy):
        itr = 0
        while True:
            if len(glob.glob(self._inference_policy_file_name(itr) + '*')) == 0:
                itr -= 1
                break
            itr += 1

        logger.info('Restore model itr {0}'.format(itr))
        if itr >= 0:
            policy.restore(self._inference_policy_file_name(itr), train=False)

    #############
    ### Files ###
    #############

    @property
    def _save_dir(self):
        return rllab_logger.get_snapshot_dir()

    def _train_policy_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_train_policy.ckpt'.format(itr))

    def _inference_policy_file_name(self, itr):
        return os.path.join(self._save_dir, 'itr_{0:04d}_inference_policy.ckpt'.format(itr))

    @property
    def _data_folder(self):
        return self._params['policy']['TraversabilityPolicy']['folder']

    @property
    def _data_train_images_file_name(self):
        return os.path.join(self._data_folder, 'data_train_images.npy')

    @property
    def _data_train_labels_file_name(self):
        return os.path.join(self._data_folder, 'data_train_labels.npy')

    @property
    def _data_holdout_images_file_name(self):
        return os.path.join(self._data_folder, 'data_holdout_images.npy')

    @property
    def _data_holdout_labels_file_name(self):
        return os.path.join(self._data_folder, 'data_holdout_labels.npy')

    def _load_train_data(self):
        images = np.expand_dims(np.load(self._data_train_images_file_name), 3)
        labels = np.load(self._data_train_labels_file_name)
        return images, labels

    def _load_holdout_data(self):
        images = np.expand_dims(np.load(self._data_holdout_images_file_name), 3)
        labels = np.load(self._data_holdout_labels_file_name)
        return images, labels

    @property
    def _offline_dir(self):
        dir = os.path.join(self._save_dir, 'eval_offline')
        os.makedirs(dir, exist_ok=True)
        return dir

    def _offline_image_file_name(self, suffix, i):
        return os.path.join(self._offline_dir, '{0}_{1:03d}.png'.format(suffix, i))

    @property
    def _online_dir(self):
        dir = os.path.join(self._save_dir, 'eval_online')
        os.makedirs(dir, exist_ok=True)
        return dir

    def _online_rollouts_file_name(self, name):
        return os.path.join(self._online_dir, '{0}.pkl'.format(name))

    def _save_online_rollouts(self, name, rollouts):
        fname = self._online_rollouts_file_name(name)
        mypickle.dump({'rollouts': rollouts}, fname)

    def _load_online_rollouts(self, name):
        fname = self._online_rollouts_file_name(name)
        return mypickle.load(fname)['rollouts']

    ##########################
    ### Offline evaluation ###
    ##########################

    def eval_offline(self):
        self._eval_offline('train', *self._load_train_data())
        self._eval_offline('holdout', *self._load_holdout_data())

    def _eval_offline(self, name, images, labels):
        ### evaluate model
        logger.info('Evaluating model')
        probs = []
        if len(images) > 10:
            indices_list = np.array_split(np.arange(len(images)), len(images) // 10)
        else:
            indices_list = [np.arange(len(images))]
        for indices in indices_list:
            probs.append(self._policy.get_model_outputs(images[indices], None))
        probs = np.vstack(probs)

        ### plot evaluation
        logger.info('Plotting evaluation')
        for i, (image, label, prob) in enumerate(zip(images, labels, probs)):
            label_coll = 1 - label
            prob_coll = prob[:, :, 0]
            pred_coll = prob.argmax(axis=2)

            if i == 0:
                # black is 0, white is 255
                f, axes = plt.subplots(1, 4, figsize=(12, 3))
                imshow0 = axes[0].imshow(image[:, :, 0], cmap='Greys_r', vmin=0, vmax=255)
                imshow1 = axes[1].imshow(label_coll, cmap='Greys_r', vmin=0, vmax=1)
                imshow2 = axes[2].imshow(prob_coll, cmap='Greys_r', vmin=0, vmax=1)
                imshow3 = axes[3].imshow(label_coll == pred_coll, cmap='Greys_r', vmin=0, vmax=1)
                for ax, title in zip(axes, ('image', 'label', 'model prob', 'error')):
                    ax.set_title(title)
                plt.pause(0.01)
            else:
                imshow0.set_data(image[:, :, 0])
                imshow1.set_data(label_coll)
                imshow2.set_data(prob_coll)
                imshow3.set_data(label_coll == pred_coll)
                f.canvas.draw()
                plt.pause(0.01)

            plt.tight_layout()
            f.savefig(self._offline_image_file_name(name, i), dpi=100, bbox_inches='tight')

        plt.close(f)

    #########################
    ### Online evaluation ###
    #########################

    def eval_online(self, name, num_rollouts):
        if not os.path.exists(self._online_rollouts_file_name(name)):
            self._generate_rollouts(name, num_rollouts)

        self._analyze_rollouts(self._load_online_rollouts(name))

    def _generate_rollouts(self, name, num_rollouts):
        logger.info('Generating rollouts')
        sampler = Sampler(policy=self._policy,
                          env=self._env,
                          n_envs=1,
                          replay_pool_size=int(1.5 * self._env.horizon),
                          max_path_length=self._env.horizon,
                          sampling_method='uniform',
                          save_rollouts=True,
                          save_rollouts_observations=True,
                          save_env_infos=True)

        sampler.reset()
        rollouts = []
        step = 0
        while len(rollouts) < num_rollouts:
            sampler.step(step, explore=False)
            rollouts += sampler.get_recent_paths()
            step += 1

        self._save_online_rollouts(name, rollouts)

    def _analyze_rollouts(self, rollouts):
        logger.info('Analyzing rollouts')

        durations = [len(r['dones']) for r in rollouts]

        logger.info('Duration: {0:.2f} +- {1:.1f}'.format(np.mean(durations), np.std(durations)))

def run_offline(params):
    eval_trav = EvalTraversabilityPolicy(params)
    eval_trav.eval_offline()

def run_online(params, name, num_rollouts):
    eval_trav = EvalTraversabilityPolicy(params)
    eval_trav.eval_online(name, num_rollouts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=('offline', 'online'))
    parser.add_argument('--exps', nargs='+')
    parser.add_argument('-name', type=str, default='test') # online only
    parser.add_argument('-numrollouts', type=int, default=10) # online only
    args = parser.parse_args()

    for exp in args.exps:
        yaml_path = os.path.abspath('/home/gkahn/code/rllab/sandbox/gkahn/gcg/yamls/{0}.yaml'.format(exp))
        assert(os.path.exists(yaml_path))
        with open(yaml_path, 'r') as f:
            params = yaml.load(f)
        with open(yaml_path, 'r') as f:
            params_txt = ''.join(f.readlines())
        params['txt'] = params_txt

        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])
        config.USE_TF = True

        rllab_logger.set_snapshot_dir(os.path.join('/home/gkahn/code/rllab/data/local',
                                                   params['exp_prefix'], params['exp_name']))

        if args.method == 'offline':
            run_offline(params)
        elif args.method == 'online':
            run_online(params, args.name, args.numrollouts)
        else:
            raise NotImplementedError
