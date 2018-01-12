import os
import argparse
import yaml
import multiprocessing
import copy

from avillaflor.gcg.algos.gcg import run_gcg
from avillaflor.gcg.utils.variants import VariantGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+')
args = parser.parse_args()

def thread_fn(q, gpu):
    while not q.empty():
        exp_params = q.get()
        exp_params['policy']['gpu_device'] = int(gpu)
        run_gcg(params)

for exp in args.exps:
    yaml_path = os.path.abspath('yamls/{0}.yaml'.format(exp))
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    with open(yaml_path, 'r') as f:
        params_txt = ''.join(f.readlines())
    params['txt'] = params_txt

    q = multiprocessing.Queue()
#    vg2 = VariantGenerator()
#    vg2.add('alpha', [1., 0.1])
#    vg2.add('multiply', ['probnocoll', None])
#    variants2 = vg2.variants()
    vg1 = VariantGenerator()
    vg1.add('fn', ['tanh', None])
    vg1.add('scale', lambda fn: [3.14159265 if fn == 'tanh' else 1.0])
    vg1.add('loss', ['mse', 'sin_2'])
    variants1 = vg1.variants()
    for seed in [0, 1, 2]:
        for variant1 in variants1:
#            for variant2 in variants2:
            exp_params = copy.deepcopy(params)
            exp_params['policy']['RCcarSensorsMACPolicy']['output_sensors'][1].update(variant1)
#            exp_params['policy']['RCcarSensorsMACPolicy']['action_value_terms'][0].update(variant2)
#            exp_params['exp_name'] = '{0}_{1}/seed_{2}'.format(vg1.to_name_suffix(variant1), vg2.to_name_suffix(variant2), seed)
            exp_params['exp_name'] = '{0}/seed_{1}'.format(vg1.to_name_suffix(variant1), seed)
            exp_params['seed'] = seed
            q.put(exp_params)

    import IPython; IPython.embed()

    num_per_gpu = int(1.0 / params['policy']['gpu_frac'])
    processes = []
    for gpu in args.gpus:
        for _ in range(num_per_gpu):
            p = multiprocessing.Process(target=thread_fn, args=(q, gpu))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
