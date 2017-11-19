import os
import argparse
import yaml

from rllab import config
from sandbox.gkahn.gcg.algos.async_gcg import run_async_gcg_train, run_async_gcg_inference
from rllab.misc.instrument import run_experiment_lite

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('async', type=str, choices=('train', 'inference'))
args = parser.parse_args()

yaml_path = os.path.abspath('yamls/{0}.yaml'.format(args.exp))
assert(os.path.exists(yaml_path))
with open(yaml_path, 'r') as f:
    params = yaml.load(f)
with open(yaml_path, 'r') as f:
    params_txt = ''.join(f.readlines())
params['txt'] = params_txt

os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])
config.USE_TF = True

if args.async == 'train':
    run_experiment_lite(
        run_async_gcg_train,
        snapshot_mode="all",
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix'],
        variant=params,
        use_gpu=True,
        use_cloudpickle=True,
    )
elif args.async == 'inference':
    run_experiment_lite(
        run_async_gcg_inference,
        snapshot_mode="all",
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix'],
        variant=params,
        use_gpu=True,
        use_cloudpickle=True,
    )