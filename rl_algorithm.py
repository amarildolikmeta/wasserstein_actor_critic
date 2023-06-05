import abc
from collections import OrderedDict
from torch import nn as nn
from utils.logging import logger
import utils.eval_util as eval_util
from utils.rng import get_global_pkg_rng_state
import utils.pytorch_util as ptu
import gtimer as gt
from replay_buffer import ReplayBuffer
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from tqdm import trange

#import ray
import torch
import numpy as np
import random

# debug: fake_policy
import utils.pytorch_util as ptu
from trainer import policies


class BatchRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_data_collector: MdpPathCollector,
            remote_eval_data_collector: RemoteMdpPathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            optimistic_exp_hp=None,
            deterministic=False,
            save_sampled_data=False,
            global_opt=False,
            save_fig=False,
            trainer_UB=False,
            ddpg=False,
            ddpg_noisy=False,
            fake_policy=False,
            random_policy=False,
            expl_policy_std=0,
            save_heatmap=False,
            domain=None,
            target_paths_qty=0,
            comp_MADE=False
    ):
        super().__init__()
        """
        The class state which should not mutate
        """
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.optimistic_exp_hp = optimistic_exp_hp
        self.deterministic = deterministic
        self.save_sampled_data = save_sampled_data
        self.ob_sampled = []
        self.ac_sampled = []
        self.global_opt = global_opt
        self.ddpg = ddpg
        self.ddpg_noisy = ddpg_noisy
        self.fake_policy = fake_policy # dev
        self.random_policy = random_policy # dev
        self.expl_policy_std = expl_policy_std
        self.save_heatmap = save_heatmap # dev
        self.domain = domain # dev
        self.target_paths_qty = target_paths_qty
        """
        The class mutable state
        """
        self._start_epoch = 0


        """
        This class sets up the main training loop, so it needs reference to other
        high level objects in the algorithm

        But these high level object maintains their own states
        and has their own responsibilities in saving and restoring their state for checkpointing
        """
        self.trainer = trainer
        self.trainer_UB = None
        if trainer_UB:
            self.trainer_UB = trainer
        self.expl_data_collector = exploration_data_collector
        self.remote_eval_data_collector = remote_eval_data_collector

        self.replay_buffer = replay_buffer
        self.save_fig = save_fig

        self.comp_MADE = comp_MADE

        if self.comp_MADE: 
            self._define_for_MADE_index()
            # replay_buffer.keep_dist_for_MADE()

        # DEBUG
        if self.random_policy:
            self.r_p = policies.UniformRandomPolicy(
                [4],
                1,
                self.trainer.policy.output_size
            )

    def _define_for_MADE_index(self):
        # only works in 1 dimension

        self.lambda_reg = 0.001
        # # self.expl_data_collector._env.
        # tot_dim = self.expl_data_collector._env.observation_space.shape[0] + \
        #     self.expl_data_collector._env.action_space.shape[0]
        
        self.histo_rho = None
        # self.histo_rho = np.zeros(shape=[n_bin] * tot_dim, dtype=np.float32) 
        # self.histo_d = np.zeros(shape=[n_bin] * tot_dim, dtype=np.float32) 

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def get_fake_policy(self, epoch):
        import json
        import os
        from utils.pythonplusplus import load_gzip_pickle
        from utils.variant_util import build_variant
        from utils.core import np_ify, torch_ify

        path = 'data/fake_policy/s1'
        variant = json.load(open(os.path.join(path, 'variant.json')))
        trainer = build_variant(variant, return_replay_buffer=False, return_collectors=False)['trainer']

        experiment = path + '/itr_' + str(epoch) + '.zip_pkl'
        exp = load_gzip_pickle(experiment)

        trainer.restore_from_snapshot(exp['trainer'])

        return trainer.policy

    def _train(self):
        
        target_steps = round(self.num_expl_steps_per_train_loop * self.target_paths_qty)
        behavioral_steps = self.num_expl_steps_per_train_loop - target_steps

        if hasattr(self.trainer, 'target_policy'):
            target_policy = self.trainer.target_policy
        else:
            target_policy = self.trainer.policy
        behavioral_policy = self.trainer.policy

        # DEBUG
        if self.random_policy:
            behavioral_policy = self.r_p

        # Fill the replay buffer to a minimum before training starts
        if self.min_num_steps_before_training > self.replay_buffer.num_steps_can_sample():
            init_expl_paths, returns = self.expl_data_collector.collect_new_paths(
                policy=behavioral_policy,
                max_path_length=self.max_path_length,
                num_steps=self.min_num_steps_before_training,
                discard_incomplete_paths=False
            )
            if self.expl_policy_std > 0:
                behavioral_policy.set_std(self.expl_policy_std) # FIXME: remove?
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        best_eval = -np.inf
        for epoch in gt.timed_for(
                trange(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # # save init
            # if epoch == 0:
            #     self._end_epoch(-1)

            # To evaluate the policy remotely,
            # we're shipping the policy params to the remote evaluator
            # This can be made more efficient
            # But this is currently extremely cheap due to small network size
            #pol_state_dict = ptu.state_dict_cpu(self.trainer.policy)

            # remote_eval_obj_id = self.remote_eval_data_collector.async_collect_new_paths.remote(
            #     self.max_path_length,
            #     self.num_eval_steps_per_epoch,
            #     discard_incomplete_paths=True,
            #     deterministic_pol=True,
            #     pol_state_dict=pol_state_dict)

            eval_exploration_paths, returns = self.remote_eval_data_collector.collect_new_paths(
                target_policy,
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                deterministic_pol=True
            )
            avg_ret = np.mean(returns)
            # if avg_ret > best_eval:
            #     snapshot = self._get_snapshot(epoch)
            #     logger.save_itr_params(epoch, snapshot, best=True)
            #     best_eval = avg_ret

            for _ in range(self.num_train_loops_per_epoch):
                deterministic = self.trainer.deterministic
                if self.expl_policy_std > 0:
                    deterministic = False
                if self.ddpg:
                    deterministic = self.ddpg_noisy
                new_expl_paths, returns = self.expl_data_collector.collect_new_paths(
                    behavioral_policy,
                    self.max_path_length,
                    behavioral_steps,
                    discard_incomplete_paths=False,
                    optimistic_exploration=self.optimistic_exp_hp['should_use'],
                    deterministic_pol=deterministic,
                    optimistic_exploration_kwargs=dict(
                        policy=behavioral_policy,
                        qfs=self.trainer.qfs,
                        trainer=self.trainer_UB,
                        hyper_params=self.optimistic_exp_hp
                    )
                )
                gt.stamp('exploration sampling', unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                if target_steps > 0:
                    new_expl_paths, returns = self.expl_data_collector.collect_new_paths(
                        target_policy,
                        self.max_path_length,
                        target_steps,
                        discard_incomplete_paths=False,
                        optimistic_exploration=self.optimistic_exp_hp['should_use'],
                        deterministic_pol=deterministic,
                        optimistic_exploration_kwargs=dict(
                            policy=behavioral_policy,
                            qfs=self.trainer.qfs,
                            trainer=self.trainer_UB,
                            hyper_params=self.optimistic_exp_hp
                        )
                    )
                #self.replay_buffer.add_paths(new_expl_paths)
                #gt.stamp('exploration sampling', unique=False)
                #self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)
                
                if hasattr(self.trainer, 'save_prv_std'):
                    self.trainer.save_prv_std()

                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    if self.save_sampled_data:
                        self.ob_sampled.append(train_data['observations'])
                        self.ac_sampled.append(train_data['actions'])
                    train_data['buffer'] = self.replay_buffer
                    self.trainer.train(train_data)

                if self.global_opt:   
                    self.trainer.optimize_policies(self.replay_buffer, out_dir=logger._snapshot_dir,
                                                  epoch=epoch, save_fig=self.save_fig)

                gt.stamp('training', unique=False)

            # Wait for eval to finish
            #ray.get([remote_eval_obj_id])
            #gt.stamp('remote evaluation wait')

            if self.fake_policy:
                fake_policy = self.get_fake_policy(epoch)
                ptu.copy_model_params_from_to(fake_policy, self.trainer.policy)

            if self.comp_MADE: 
                self._compute_MADE(new_expl_paths, epoch)
    
            self._end_epoch(epoch)
        eval_exploration_paths = self.remote_eval_data_collector.collect_new_paths(
            target_policy,
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
            deterministic_pol=True
        )

    def _compute_MADE(self, paths, epoch):
        epoch += 1
        array_path_o = np.zeros([self.num_expl_steps_per_train_loop])
        array_path_a = np.zeros([self.num_expl_steps_per_train_loop])
        head = 0
        for path in paths:
            for i, (
                    obs,
                    action,
                ) in enumerate(zip(
                    path["observations"],
                    path["actions"],
                )):
                
                array_path_o[head] = obs
                array_path_a[head] = action
                head += 1
            
        self.histo_d = np.histogram2d(array_path_o, array_path_a, bins=[20,20], range=[[-1, 1], [-1, 1]], density=True)[0]  

        if self.histo_rho is None:
            self.histo_rho = np.zeros_like(self.histo_d)

        self.MADE = np.sum(np.sqrt((self.histo_d + self.lambda_reg)/(self.histo_rho + self.lambda_reg)))

        self.histo_rho = self.histo_rho * ((epoch - 1) / epoch)
        self.histo_rho += self.histo_d / epoch


    def _end_epoch(self, epoch):
        if epoch == -1: 
            logger.save_heatmap(self.trainer, self.domain, epoch, [], [])
            return

        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        #ray.get([self.remote_eval_data_collector.end_epoch(epoch)])
        self.remote_eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        # We can only save the state of the program
        # after we call end epoch on all objects with internal state.
        # This is so that restoring from the saved state will
        # lead to identical result as if the program was left running.
        if epoch >= 0:
            snapshot = self._get_snapshot(epoch)
            logger.save_itr_params(epoch, snapshot)
            #gt.stamp('saving')

        if self.save_sampled_data:
            state_data = np.concatenate(self.ob_sampled)
            action_data = np.concatenate(self.ac_sampled)
            # if not self.save_heatmap:
            logger.save_sampled_data(state_data, action_data)
            del self.ob_sampled
            del self.ac_sampled
            self.ob_sampled = []
            self.ac_sampled = []
            
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)

        write_header = True if epoch == 0 else False
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)

        if self.save_heatmap and self.save_sampled_data:
            logger.save_heatmap(self.trainer, self.domain, epoch, state_data, action_data)
        elif self.save_heatmap:
            logger.save_heatmap(self.trainer, self.domain, epoch, None, None)

        # self.expl_data_collector._env.save_bounds_file()
        # self.remote_eval_data_collector._env.save_bounds_file()

    def _get_snapshot(self, epoch):
        snapshot = dict(
            trainer=self.trainer.get_snapshot(),
            exploration=self.expl_data_collector.get_snapshot(),
            evaluation_remote=self.remote_eval_data_collector.get_snapshot(),
            # evaluation_remote_rng_state=ray.get(
            #     self.remote_eval_data_collector.get_global_pkg_rng_state.remote()
            # ),
            replay_buffer=self.replay_buffer.get_snapshot()
        )

        # What epoch indicates is that at the end of this epoch,
        # The state of the program is snapshot
        # Not to be consfused with at the beginning of the epoch
        snapshot['epoch'] = epoch

        # Save the state of various rng
        snapshot['global_pkg_rng_state'] = get_global_pkg_rng_state()

        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Remote Evaluation
        """
        logger.record_dict(self.remote_eval_data_collector.get_diagnostics(),
            prefix='remote_evaluation/',
        )
        remote_eval_paths = self.remote_eval_data_collector.get_epoch_paths()
        logger.record_dict(
            eval_util.get_generic_path_information(remote_eval_paths),
            prefix="remote_evaluation/",
        )
       
        """
        Scores
        """
        if self.comp_MADE:
            made_dict = OrderedDict()
            made_dict['MADE'] = self.MADE
            logger.record_dict(
                made_dict,
                prefix='scores/'
            )

        """
        Misc
        """
        gt.stamp('logging')

    def to(self, device):
        #print(self.trainer.networks)
        for net in self.trainer.networks:
            #print(net)
            net.to(device)


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times
