import os
import argparse
import yaml

from sandbox.gkahn.algos.rw_rccar_offline import run_rw_rccar_offline

parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+')
args = parser.parse_args()

curr_dir = os.path.realpath(os.path.dirname(__file__))
yaml_dir = os.path.join(curr_dir[:curr_dir.find('src/sandbox')], 'yamls')

for exp in args.exps:
    yaml_path = os.path.join(yaml_dir, '{0}.yaml'.format(exp))
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    with open(yaml_path, 'r') as f:
        params_txt = ''.join(f.readlines())
    params['txt'] = params_txt

    run_rw_rccar_offline(params)