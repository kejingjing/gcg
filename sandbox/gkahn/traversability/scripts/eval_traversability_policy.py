import argparse, os, glob
import yaml

import numpy as np
import matplotlib.pyplot as plt

from rllab import config
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

    def _online_dir(self, name):
        dir = os.path.join(self._save_dir, 'eval_online/{0}'.format(name))
        os.makedirs(dir, exist_ok=True)
        return dir

    def _online_rollouts_file_name(self, name):
        return os.path.join(self._online_dir(name), 'rollouts.pkl')

    def _save_online_rollouts(self, name, rollouts):
        fname = self._online_rollouts_file_name(name)
        mypickle.dump({'rollouts': rollouts}, fname)

    def _load_online_rollouts(self, name):
        fname = self._online_rollouts_file_name(name)
        return mypickle.load(fname)['rollouts']

    def _online_rollouts_images_file_name(self, name, rollout_num, timestep):
        return os.path.join(self._online_dir(name), 'r{0:02d}_t{1:04}.png'.format(rollout_num, timestep))

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
            label_nocoll = 1 - label
            prob_nocoll = prob[:, :, 0]
            pred_nocoll = prob.argmax(axis=2)

            if i == 0:
                # black is 0, white is 255
                f, axes = plt.subplots(1, 4, figsize=(12, 3))
                imshow0 = axes[0].imshow(image[:, :, 0], cmap='Greys_r', vmin=0, vmax=255)
                imshow1 = axes[1].imshow(label_nocoll, cmap='Greys_r', vmin=0, vmax=1)
                imshow2 = axes[2].imshow(prob_nocoll, cmap='Greys_r', vmin=0, vmax=1)
                imshow3 = axes[3].imshow(label_nocoll == pred_nocoll, cmap='Greys_r', vmin=0, vmax=1)
                for ax, title in zip(axes, ('image', 'label', 'model prob', 'error')):
                    ax.set_title(title)
                plt.pause(0.01)
            else:
                imshow0.set_data(image[:, :, 0])
                imshow1.set_data(label_nocoll)
                imshow2.set_data(prob_nocoll)
                imshow3.set_data(label_nocoll == pred_nocoll)
                f.canvas.draw()
                plt.pause(0.01)

            plt.tight_layout()
            f.savefig(self._offline_image_file_name(name, i), dpi=100, bbox_inches='tight')

        plt.close(f)

    #########################
    ### Online evaluation ###
    #########################

    def eval_online(self, name, num_rollouts, get_steer_func=None):
        if not os.path.exists(self._online_rollouts_file_name(name)):
            self._generate_rollouts(name, num_rollouts, get_steer_func)

        logger.info(name)
        self._analyze_rollouts(name)
        logger.info('')

    def _generate_rollouts(self, name, num_rollouts, get_steer_func):
        self._policy.set_get_steer(get_steer_func)

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

    def _analyze_rollouts(self, name):
        rollouts = self._load_online_rollouts(name)
        durations = [len(r['dones']) for r in rollouts]

        logger.info('Duration: {0:.2f} +- {1:.1f}'.format(np.mean(durations), np.std(durations)))

        create_images = not os.path.exists(self._online_rollouts_images_file_name(name, 0, 0))
        if create_images:
            for i, r in enumerate(rollouts):
                for j, (obs, action, env_info) in enumerate(zip(r['observations'], r['actions'], r['env_infos'])):
                    im = np.reshape(obs, (36, 64))
                    probcoll = env_info['prob_coll']

                    angle = -(np.pi / 2.) * action[0]
                    xstart = 64 / 2.
                    ystart = 34.
                    dx = 10. * np.sin(angle)
                    dy = -10. * np.cos(angle)

                    if i == 0 and j == 0:
                        f, axes = plt.subplots(1, 2, figsize=(12, 6))
                        imshow = axes[0].imshow(im, cmap='Greys_r')
                        collshow = axes[1].imshow(1 - probcoll, cmap='Greys_r', vmin=0, vmax=1)
                        plt.tight_layout()
                    else:
                        imshow.set_data(im)
                        collshow.set_data(1 - probcoll)
                        arrow0.remove()
                        arrow1.remove()

                    arrow0 = axes[0].arrow(xstart, ystart, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
                    arrow1 = axes[1].arrow(xstart, ystart, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')
                    for ax in axes:
                        ax.set_title('{0:.2f}'.format(action[0]))

                    f.savefig(self._online_rollouts_images_file_name(name, i, j), bbox_inches='tight', dpi=100)

            plt.close(f)

def run_offline(params):
    eval_trav = EvalTraversabilityPolicy(params)
    eval_trav.eval_offline()

def run_online(params):
    eval_trav = EvalTraversabilityPolicy(params)
    num_rollouts = 4

    dict_list = []

    ### Find costs of left and right half of image and use e.g. ratio to calculate angle ###
    def get_steer_func(kp):
        def func(prob_coll):
            obs_shape = list(prob_coll.shape)

            weights = np.ones(obs_shape, dtype=np.float32)

            cost_map = prob_coll * weights
            cost_left = cost_map[:, :obs_shape[1]//2].sum()
            cost_right = cost_map[:, obs_shape[1]//2:].sum()

            steer = kp * 2. * (cost_right / (cost_left + cost_right) - 0.5)
            return steer

        return func

    for kp in [5.0, 15.0]:
        d = {
            'name': 'halves_kp_{0:.2f}'.format(kp),
            'get_steer_func': get_steer_func(kp)
        }
        dict_list.append(d)

    ### Find highest point that is collision free and use as set point ###

    def get_steer_func(coll_buffer, kp):
        def func(prob_coll):
            obs_shape = list(prob_coll.shape)

            ### binarize
            prob_coll = (prob_coll > 0.5).astype(np.float32)

            ### apply buffer
            import scipy.ndimage.filters
            prob_coll_buffered = scipy.ndimage.filters.maximum_filter(prob_coll, coll_buffer + 1)

            ### index of column with tallest continuous no collision
            steer_index = (np.flipud(prob_coll_buffered).cumsum(axis=0) > 0).argmax(axis=0).argmax()

            steer = -kp * (steer_index - (obs_shape[1]-1) / 2.) / float(obs_shape[1] / 2.)
            return steer
        return func

    for coll_buffer in [2]: #range(0, 10, 2):
        for kp in [1.]: #np.linspace(0.05, 1.0, 5):
            d = {
                'name': 'buffer_{0:d}_kp_{1:.2f}'.format(coll_buffer, kp),
                'get_steer_func': get_steer_func(coll_buffer, kp)
            }
            dict_list.append(d)

    for d in dict_list:
        eval_trav.eval_online(d['name'], num_rollouts, d['get_steer_func'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=('offline', 'online'))
    parser.add_argument('--exps', nargs='+')
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
            run_online(params)
        else:
            raise NotImplementedError
