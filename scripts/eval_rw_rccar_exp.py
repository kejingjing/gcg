import os
import argparse
import yaml

from gcg.algos.eval_rw_rccar_gcg import eval_rw_rccar_gcg

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str)
parser.add_argument('-itr', type=int)
args = parser.parse_args()

yaml_path = os.path.abspath('../yamls/{0}.yaml'.format(args.exp))
assert(os.path.exists(yaml_path))
with open(yaml_path, 'r') as f:
    params = yaml.load(f)
with open(yaml_path, 'r') as f:
    params_txt = ''.join(f.readlines())
params['txt'] = params_txt

eval_rw_rccar_gcg(params, args.itr)
