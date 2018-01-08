import argparse, os, glob
import yaml

import numpy as np
import matplotlib.pyplot as plt

from rllab import config
from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as rllab_logger

from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.envs.env_utils import create_env

def run(params):
    logger.setup_logger(os.path.join(rllab_logger.get_snapshot_dir(), 'log_eval.txt'),
                        params['log_level'])

    env = create_env(params['alg']['env'], is_normalize=params['alg']['normalize_env'])

    ### setup policy
    logger.info('Setting up policy')
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

    ### load data
    logger.info('Loading data')
    folder = policy_params['folder']
    images = np.expand_dims(np.load(os.path.join(folder, 'data_images.npy')), 3)
    labels = np.load(os.path.join(folder, 'data_labels.npy'))

    ### load model
    logger.info('Loading model')
    save_dir = rllab_logger.get_snapshot_dir()
    def train_policy_file_name(itr):
        return os.path.join(save_dir, 'itr_{0:04d}_train_policy.ckpt'.format(itr))
    def inference_policy_file_name(itr):
        return os.path.join(save_dir, 'itr_{0:04d}_inference_policy.ckpt'.format(itr))

    itr = 0
    while True:
        if len(glob.glob(inference_policy_file_name(itr) + '*')) == 0:
            itr -= 1
            break
        itr += 1

    logger.info('Restore model itr {0}'.format(itr))
    if itr >= 0:
        policy.restore(inference_policy_file_name(itr), train=False)

    ### evaluate model
    logger.info('Evaluating model')
    probs = []
    indices_list = np.array_split(np.arange(len(images)), len(images) // 10)
    for indices in indices_list:
        probs.append(policy.get_model_outputs(images[indices], None))
    probs = np.vstack(probs)

    ### plot evaluation
    logger.info('Plotting evaluation')
    for i, (image, label, prob) in enumerate(zip(images, labels, probs)):
        label_coll = 1 - label
        prob_coll = prob[:,:,0]
        pred_coll = prob.argmax(axis=2)

        if i == 0:
            # black is 0, white is 255
            f, axes = plt.subplots(1, 4, figsize=(12, 3))
            imshow0 = axes[0].imshow(image[:,:,0], cmap='Greys_r', vmin=0, vmax=255)
            imshow1 = axes[1].imshow(label_coll, cmap='Greys_r', vmin=0, vmax=1)
            imshow2 = axes[2].imshow(prob_coll, cmap='Greys_r', vmin=0, vmax=1)
            imshow3 = axes[3].imshow(label_coll == pred_coll, cmap='Greys_r', vmin=0, vmax=1)
            for ax, title in zip(axes, ('image', 'label', 'model prob', 'error')):
                ax.set_title(title)
            plt.pause(0.01)
        else:
            imshow0.set_data(image[:,:,0])
            imshow1.set_data(label_coll)
            imshow2.set_data(prob_coll)
            imshow3.set_data(label_coll == pred_coll)
            f.canvas.draw()
            plt.pause(0.01)

        plt.tight_layout()
        f.savefig(os.path.join(save_dir, 'eval{0:03d}.png'.format(i)), dpi=100, bbox_inches='tight')

    plt.close(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

        run_experiment_lite(
            run,
            snapshot_mode="all",
            exp_name=params['exp_name'],
            exp_prefix=params['exp_prefix'],
            variant=params,
            use_gpu=True,
            use_cloudpickle=True,
        )
