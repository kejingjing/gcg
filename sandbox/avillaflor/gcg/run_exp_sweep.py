import os, time
import argparse
import yaml
import multiprocessing

from rllab import config
from sandbox.avillaflor.gcg.algos.gcg import run_gcg
from rllab.misc.instrument import stub, run_experiment_lite, VariantGenerator
import rllab.misc.logger as logger

parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+')
parser.add_argument('-mode', type=str, default='local')
parser.add_argument('--confirm_remote', action='store_false')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--gpus', type=list, default=[])
parser.add_argument('-region', type=str, choices=('us-west-1', 'us-west-2', 'us-east-1', 'us-east-2'), default='us-west-1')
args = parser.parse_args()


# TODO instead use process per open gpu and then queue of parameters
def thread_fn(exp_params, gpus, usage, sem, lock):
    sem.acquire()
    lock.acquire()
    for i, gpu in enumerate(gpus):
        if usage[i] > 0:
            exp_params['policy']['gpu_device'] = int(gpu)
            usage[i] -= 1
            break
    lock.release()
    run_experiment_lite(
        run_gcg,
        snapshot_mode="all",
        exp_name=exp_params['exp_name'],
        exp_prefix=exp_params['exp_prefix'],
        variant=exp_params,
        use_gpu=True,
        use_cloudpickle=True,
    )
    lock.acquire()
    for i, gpu in enumerate(gpus):
        if gpu == exp_params['policy']['gpu_device']:
            usage[i] += 1
            break
    lock.release()
    sem.release()

for exp in args.exps:
    yaml_path = os.path.abspath('yamls/{0}.yaml'.format(exp))
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    with open(yaml_path, 'r') as f:
        params_txt = ''.join(f.readlines())
    params['txt'] = params_txt

    num_per_gpu = int(1.0 / params['policy']['gpu_frac'])
    usage = multiprocessing.Array('i', [num_per_gpu] * len(args.gpus))
    gpus = multiprocessing.Array('c', str.encode(''.join(args.gpus)))
#    gpus = []
#    for i in args.gpus:
#        gpus.append([i, num_per_gpu])
    
    if len(gpus) * num_per_gpu > 1:
        logger.set_log_tabular_only(True)
    
    # for multiprocessing
    sem = multiprocessing.Semaphore(len(gpus) * num_per_gpu)
    lock = multiprocessing.Lock()
    processes = []

    vg = VariantGenerator()
    vg.add('fn', ['tanh', None])
    vg.add('scale', lambda fn: [3.14159265 if fn == 'tanh' else 1.0])
    vg.add('cum_sum', ['post', None])
    vg.add('loss', ['mse', 'sin_2'])
    variants = vg.variants()
    for seed in [0, 1, 2]:
        for variant in variants:
            exp_params = {**params}
            exp_params['policy']['RCcarSensorsMACPolicy']['output_sensors'][1].update(variant)
#            exp_params = {**params, **variant}
#            exp_params['exp_prefix'] ='{0}_{1}_seed_{2}'.format(vg.to_name_suffix(variant), exp_params['exp_prefix'], seed)
            exp_params['exp_name'] ='{0}_seed_{1}'.format(vg.to_name_suffix(variant), seed)
            exp_params['seed'] = seed
            p = multiprocessing.Process(target=thread_fn, args=(exp_params, gpus, usage, sem, lock))
            p.start()
            processes.append(p)
#        import IPython; IPython.embed()
    # TODO
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])
#        config.USE_TF = True
#
#        run_experiment_lite(
#            run_gcg,
#            snapshot_mode="all",
#            exp_name=exp_params['exp_name'],
#            exp_prefix=exp_params['exp_prefix'],
#            variant=exp_params,
#            use_gpu=True,
#            use_cloudpickle=True,
#        )
    for p in processes:
        p.join()
