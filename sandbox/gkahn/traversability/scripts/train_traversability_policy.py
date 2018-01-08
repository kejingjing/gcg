import argparse, os
import yaml

import numpy as np

from rllab import config
from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as rllab_logger

from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.envs.env_utils import create_env

def run(params):
    logger.setup_logger(os.path.join(rllab_logger.get_snapshot_dir(), 'log.txt'),
                        params['log_level'])

    env = create_env(params['alg']['env'], is_normalize=params['alg']['normalize_env'])

    ### setup policy
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
    folder = policy_params['folder']
    images = np.expand_dims(np.load(os.path.join(folder, 'data_images.npy')), 3)
    labels = np.load(os.path.join(folder, 'data_labels.npy'))

    for step in range(int(params['policy']['train_steps'])):
        indices = np.random.randint(low=0, high=len(images), size=params['alg']['batch_size'])

        images_batch = images[indices]
        labels_batch = labels[indices]

        policy.train_step_offline(images_batch, labels_batch)

        if step > 0 and step % int(params['alg']['log_every_n_steps']) == 0:
            policy.log()

        # TODO: save params

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
