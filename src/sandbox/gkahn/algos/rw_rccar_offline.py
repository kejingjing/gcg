import os
from collections import defaultdict

import numpy as np

from gcg.envs.env_utils import create_env
from gcg.policies.gcg_policy import GCGPolicy
from gcg.policies.probcoll_gcg_policy import ProbcollGCGPolicy
from gcg.algos.gcg import GCG
from gcg.sampler.sampler import Sampler
from gcg.data.logger import logger
from gcg.data import mypickle

import rosbag

class RWrccarOffline(GCG):

    def __init__(self, **kwargs):
        self._save_dir = kwargs['save_dir']
        self._env = kwargs['env']
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

        def visualize(rollout):
            r_len = len(rollout['dones'])
            import matplotlib.pyplot as plt

            f, ax = plt.subplots(1, 1)

            imshow = ax.imshow(rollout['observations'][0][0][:, :, 0], cmap='Greys_r')
            ax.set_title('r: {0}'.format(rollout['rewards'][0]))
            plt.show(block=False)
            plt.pause(0.01)
            input('t: 0')
            for t in range(1, r_len):
                imshow.set_data(rollout['observations'][t][0][:, :, 0])
                ax.set_title('r: {0}'.format(rollout['rewards'][t]))
                f.canvas.draw()
                plt.pause(0.01)
                input('t: {0}'.format(t))

            plt.close(f)

        for folder_num, folder in enumerate(sorted(folders)):
            assert (os.path.exists(folder))
            logger.info('Loading offpolicy rosbag data from {0}'.format(folder))

            pkl_fname = os.path.join(self._save_dir, 'rosbag_rollouts_{0:02d}.pkl'.format(folder_num))
            timesteps_kept = 0
            timesteps_total = 0
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

                    timesteps_total += len(d_bag['mode'])

                    ### trim to whenever collision occurs
                    colls = np.array([msg.data for msg in d_bag['collision/all']])
                    if len(colls) == 0:
                        logger.warn('{0}: empty bag'.format(os.path.basename(fname)))
                        continue

                    def prune(start, end):
                        for key, value in d_bag.items():
                            d_bag[key] = value[start:end]

                    ### prune based on collisions
                    if colls.max() > 0:
                        prune(None, colls.argmax() + 1)

                    encoders = np.array([msg.data for msg in d_bag['encoder/both']])
                    ### make sure it moved at least a little bit
                    if (encoders > 1e-4).sum() < 2:
                        logger.warn('{0}: car never moved'.format(os.path.basename(fname)))
                        continue
                    ### trim out initial negative encoders
                    if encoders.min() < 0:
                        start_idx = len(encoders) - (encoders[::-1] < 0).argmax()
                        prune(start_idx, None)

                    ### prune based on when collision goes zero
                    encoders = np.array([msg.data for msg in d_bag['encoder/both']])
                    is_encoder_collision = False
                    for t in range(len(encoders)-1, 0, -1):
                        if encoders[t-1] > 0 and abs(encoders[t]) < 1e-4:
                            is_encoder_collision = True
                            prune(None, t+1)
                            break

                    bag_length = len(d_bag['mode'])
                    if bag_length < 2:
                        logger.warn('{1}: length {0}'.format(bag_length, os.path.basename(fname)))
                        continue

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
                                self._env.ros_msg_update(d_bag[key][t], [key])
                            except:
                                import IPython; IPython.embed()

                    rollout = defaultdict(list)
                    update_env(0)
                    curr_obs = self._env.reset(offline=True)
                    for t in range(1, bag_length):
                        update_env(t)
                        action = [d_bag['cmd/steer'][t-1].data, d_bag['cmd/vel'][t-1].data]
                        next_obs, reward, done, _ = self._env.step(action, offline=True)

                        rollout['observations'].append(curr_obs)
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

                    ### do some post-processing to deal with bag errors
                    rollout['dones'][-1] = True
                    if is_encoder_collision:
                        rollout['rewards'][-1] = -1.

                    if len(rollout['dones']) > 1:
                        assert((rollout['rewards'][:-1] == 0).all())
                    assert(rollout['rewards'][-1] == 0 or rollout['rewards'][-1] == -1)

                    if len(rollout['dones']) < self._env.horizon - 1 and rollout['rewards'][-1] != -1:
                        logger.warn('{0}: rollout is not horizon length and does not end in collision'.format(os.path.basename(fname)))
                        continue

                    timesteps_kept += len(rollout['dones'])
                    rollouts.append(rollout)

                logger.info('Saving {0:e} timesteps ({1:.2f} kept)'.format(timesteps_kept, timesteps_kept / float(timesteps_total)))
                logger.info('Saving pkl {0}'.format(pkl_fname))
                mypickle.dump({'rollouts': rollouts}, pkl_fname)

            logger.info('Loading pkl {0}'.format(pkl_fname))
            self._sampler.add_rollouts([pkl_fname])

    ########################
    ### Training methods ###
    ########################

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
                logger.record_tabular('Step', step)
                self._policy.log()
                logger.dump_tabular(print_func=logger.info)

            ### save model
            if step > 0 and step % self._save_every_n_steps == 0:
                logger.info('Saving files for itr {0}'.format(save_itr))
                self._save_train(save_itr)
                save_itr += 1

        self._save_train(save_itr)

def run_rw_rccar_offline(params):
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir[:curr_dir.find('src/sandbox')], 'data')
    assert (os.path.exists(data_dir))
    save_dir = os.path.join(data_dir, params['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    logger.setup(display_name=params['exp_name'],
                 log_path=os.path.join(save_dir, 'log.txt'),
                 lvl=params['log_level'])

    # TODO: set seed

    # copy yaml for posterity
    yaml_path = os.path.join(save_dir, 'params.yaml'.format(params['exp_name']))
    with open(yaml_path, 'w') as f:
        f.write(params['txt'])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU

    env_str = params['alg'].pop('env')
    env = create_env(env_str, seed=params['seed'])

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
        save_dir=save_dir,
        env=env,
        policy=policy,
        max_path_length=max_path_length,
        batch_size=params['alg']['batch_size'],
        **params['alg']['offpolicy_rosbags']
    )
    algo.train()