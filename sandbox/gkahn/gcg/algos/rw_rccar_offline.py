import os
from collections import defaultdict

import numpy as np

from rllab.misc.overrides import overrides
import rllab.misc.logger as rllab_logger
from rllab import config

from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.policies.mac_policy import MACPolicy
from sandbox.gkahn.gcg.policies.rccar_mac_policy import RCcarMACPolicy
from sandbox.gkahn.gcg.algos.gcg import GCG
from sandbox.gkahn.gcg.sampler.sampler import Sampler
from sandbox.gkahn.gcg.utils import logger
from sandbox.gkahn.gcg.utils import utils
from sandbox.gkahn.gcg.utils import mypickle

import rosbag

class RWrccarOffline(GCG):

    def __init__(self, **kwargs):
        self._env = kwargs['env']
        self._inner_env = utils.inner_env(self._env)
        self._policy = kwargs['policy']

        self._batch_size = kwargs['batch_size']

        self._sampler = Sampler(
            policy=self._policy,
            env=self._env,
            n_envs=1,
            replay_pool_size=int(1e6),
            max_path_length=kwargs['max_path_length'],
            sampling_method='uniform',
            save_rollouts=False,
            save_rollouts_observations=False,
            save_env_infos=False,
            env_str=None,
            replay_pool_params={}
        )

        self._total_steps = int(kwargs['total_steps'])
        self._save_every_n_steps = int(kwargs['save_every_n_steps'])
        self._update_target_every_n_steps = int(kwargs['update_target_every_n_steps'])
        self._log_every_n_steps = int(kwargs['log_every_n_steps'])

        rosbag_folders = kwargs['folders']
        assert(len(rosbag_folders) > 0)
        self._add_offpolicy_rosbags(rosbag_folders)
        logger.info('Loaded {0} steps'.format(len(self._sampler)))

    ###############
    ### Restore ###
    ###############

    def _add_offpolicy_rosbags(self, folders):
        """ Convert rosbags to pkls and save """
        for folder_num, folder in enumerate(sorted(folders)):
            assert (os.path.exists(folder))
            logger.info('Loading offpolicy rosbag data from {0}'.format(folder))

            pkl_fname = os.path.join(self._save_dir, 'rosbag_rollouts_{0:02d}.pkl'.format(folder_num))

            if not os.path.exists(pkl_fname):
                rollouts = []
                rosbag_fnames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if '.bag' in f])
                logger.info('Reading rosbag folder {0}'.format(folder))
                for fname in rosbag_fnames:

                    ### read bag file
                    try:
                        bag = rosbag.Bag(fname, 'r')
                    except:
                        logger.warn('{0}: could not open'.format(os.path.basename(fname)))
                        continue
                    d_bag = defaultdict(list)
                    for topic, msg, t in bag.read_messages():
                       d_bag[topic].append(msg)
                    bag.close()

                    ### trim to whenever collision occurs
                    colls = [msg.data for msg in d_bag['collision/all']]
                    if len(colls) == 0:
                        logger.warn('{0}: empty bag'.format(os.path.basename(fname)))
                        continue
                    if max(colls) == 0:
                        bag_length = len(colls)
                    else:
                        bag_length = np.argmax(colls) + 1
                    if bag_length < 2:
                        logger.warn('{1}: length {0}'.format(bag_length, os.path.basename(fname)))
                        continue
                    for key, value in d_bag.items():
                        d_bag[key] = value[:bag_length]

                    ### some commands might be off by one, just append them
                    good_cmds = True
                    for key, value in d_bag.items():
                        if len(value) == bag_length:
                            pass
                        elif len(value) == bag_length - 1:
                            value.append(value[-1])
                        else:
                            logger.warn('{0}: bad command length'.format(os.path.basename(fname)))
                            good_cmds = False
                            break

                    if not good_cmds:
                        continue

                    ### update env and step

                    def update_env(t):
                        for key in d_bag.keys():
                            try:
                                self._inner_env.ros_msg_update(d_bag[key][t], [key])
                            except:
                                import IPython; IPython.embed()

                    rollout = defaultdict(list)
                    update_env(0)
                    curr_obs = self._inner_env.reset(offline=True)
                    for t in range(1, bag_length):
                        update_env(t)
                        action = [d_bag['cmd/steer'][t-1].data, d_bag['cmd/vel'][t-1].data]
                        next_obs, reward, done, _ = self._inner_env.step(action, offline=True)

                        rollout['observations'].append(curr_obs.ravel())
                        rollout['actions'].append(action)
                        rollout['rewards'].append(reward)
                        rollout['dones'].append(done or (t == bag_length - 1))
                        rollout['est_values'].append(np.nan)
                        rollout['logprobs'].append(np.nan)

                        # if done:
                        #     # assert (t == bag_length - 1)
                        #     if t != bag_length - 1:
                        #         logger.warn('done at {0} instead of {1}, bag {2}'.format(t, bag_length-1, os.path.basename(fname)))
                        #         import IPython; IPython.embed()
                        #     break
                        curr_obs = next_obs

                    for key, value in rollout.items():
                        rollout[key] = np.array(value)

                    rollouts.append(rollout)

                logger.info('Saving pkl {0}'.format(pkl_fname))
                mypickle.dump({'rollouts': rollouts}, pkl_fname)

            logger.info('Loading pkl {0}'.format(pkl_fname))
            self._sampler.add_rollouts([pkl_fname])

    ########################
    ### Training methods ###
    ########################

    @overrides
    def train(self):
        ### restore where we left off
        save_itr = self._restore()

        target_updated = False

        self._policy.update_preprocess(self._sampler.statistics)

        for step in range(0, self._total_steps):
            ### training step
            batch = self._sampler.sample(self._batch_size)
            self._policy.train_step(step, *batch, use_target=target_updated)

            ### update target network
            if step > 0 and step % self._update_target_every_n_steps == 0:
                self._policy.update_target()
                target_updated = True

            ### log
            if step % self._log_every_n_steps == 0:
                logger.info('step %.3e' % step)
                rllab_logger.record_tabular('Step', step)
                self._policy.log()
                rllab_logger.dump_tabular(with_prefix=False)

            ### save model
            if step > 0 and step % self._save_every_n_steps == 0:
                logger.info('Saving files for itr {0}'.format(save_itr))
                self._save_train(save_itr)
                save_itr += 1

        self._save_train(save_itr)

def run_rw_rccar_offline(params):
    logger.setup_logger(os.path.join(rllab_logger.get_snapshot_dir(), 'log.txt'),
                        params['log_level'])

    # copy yaml for posterity
    try:
        yaml_path = os.path.join(rllab_logger.get_snapshot_dir(), '{0}.yaml'.format(params['exp_name']))
        with open(yaml_path, 'w') as f:
            f.write(params['txt'])
    except:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU
    config.USE_TF = True

    normalize_env = params['alg'].pop('normalize_env')

    env_str = params['alg'].pop('env')
    env = create_env(env_str, is_normalize=normalize_env, seed=params['seed'])

    #####################
    ### Create policy ###
    #####################

    policy_class = params['policy']['class']
    PolicyClass = eval(policy_class)
    policy_params = params['policy'][policy_class]

    policy = PolicyClass(
        env_spec=env.spec,
        exploration_strategies=params['alg'].pop('exploration_strategies'),
        **policy_params,
        **params['policy']
    )

    ########################
    ### Create algorithm ###
    ########################

    if 'max_path_length' in params['alg']:
        max_path_length = params['alg'].pop('max_path_length')
    else:
        max_path_length = env.horizon
    algo = RWrccarOffline(
        env=env,
        policy=policy,
        max_path_length=max_path_length,
        batch_size=params['alg']['batch_size'],
        **params['alg']['offpolicy_rosbags']
    )
    algo.train()
