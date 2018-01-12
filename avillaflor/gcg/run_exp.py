import os
import argparse
import yaml

from avillaflor.gcg.algos.gcg import run_gcg

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
    params['exp_name'] = '{0}/seed_{1}'.format(params['exp_name'], params['seed'])

    run_gcg(params)
