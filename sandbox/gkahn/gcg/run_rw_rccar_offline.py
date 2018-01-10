import os
import argparse
import yaml

from rllab import config
from sandbox.gkahn.gcg.algos.rw_rccar_offline import run_rw_rccar_offline
from rllab.misc.instrument import run_experiment_lite

parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+')
args = parser.parse_args()

for exp in args.exps:
    yaml_path = os.path.abspath('yamls/{0}.yaml'.format(exp))
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    with open(yaml_path, 'r') as f:
        params_txt = ''.join(f.readlines())
    params['txt'] = params_txt

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])
    config.USE_TF = True

    run_experiment_lite(
        run_rw_rccar_offline,
        snapshot_mode="all",
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix'],
        variant=params,
        use_gpu=True,
        use_cloudpickle=True,
    )
