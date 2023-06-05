from collections import OrderedDict
from email import policy
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from utils.core import grad_norm, np_to_pytorch_batch

import utils.pytorch_util as ptu
from trainer.trainer import SACTrainer
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
from utils.core import torch_ify

class GaussianTrainerOAC(SACTrainer):
    def __init__(
            self,
            policy_producer,
            q_producer,
            action_space=None,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            qf_lr=1e-3,
            std_lr=3e-5,
            optimizer_class=optim.Adam,
            policy_weight_decay=0,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            deterministic=False,
            q_min=0,
            q_max=100,
            counts=False,
            std_soft_update=False,
            std_soft_update_prob=0.,
            train_bias=True,
            prv_std_qty=0,
            prv_std_weight=1,
            dont_use_target_std=False,
    ):
        super().__init__(
            policy_producer,
            q_producer,
            action_space=action_space,
            discount=discount,
            reward_scale=reward_scale,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            optimizer_class=optimizer_class,
            policy_weight_decay=policy_weight_decay,
            soft_target_tau=soft_target_tau,
            target_update_period=target_update_period,
            use_automatic_entropy_tuning=use_automatic_entropy_tuning,
            target_entropy=target_entropy,
            deterministic=deterministic
        )

        std = (q_max - q_min) / np.sqrt(12)
        log_std = np.log(std)
        self.std_init = std
        # self.qfs = []
        #self.qf_optimizers = []
        #self.tfs = []
        self.counts = counts
        self.prv_std_qty = prv_std_qty
        self.prv_std_weight = prv_std_weight

        self.std_soft_update = std_soft_update
        self.std_soft_update_prob = std_soft_update_prob
        self.use_target_std = not dont_use_target_std

        self.std = q_producer(bias=log_std, positive=True, train_bias=train_bias)
        self.std_target = q_producer(bias=log_std, positive=True, train_bias=train_bias)
        self.std_optimizer = optimizer_class(
            self.std.parameters(),
            lr=std_lr, )

        if self.prv_std_qty > 0:
            self.prv_std = q_producer(bias=log_std, positive=True) 
            ptu.copy_model_params_from_to(self.std, self.prv_std)


    def train(self, np_batch):
        buffer = np_batch.pop('buffer', None)
        batch = np_to_pytorch_batch(np_batch)
        batch['buffer'] = buffer
        self.train_from_torch(batch)

    def predict(self, obs, action,  upper_bound=True, beta_UB=4.66, both_values=False, target=False):
        if isinstance(obs, np.ndarray):
            obs = torch_ify(obs)
            action = torch_ify(action)
        if len(obs.shape) == 1:
            args = list(torch.unsqueeze(i, dim=0) for i in (obs, action))
        else:
            args = list(i for i in (obs, action))
        if not target:
            Q1 = self.qfs[0](*args)
            Q2 = self.qfs[1](*args)
        else:
            Q1 = self.tfs[0](*args)
            Q2 = self.tfs[1](*args)
        mu_Q = (Q1 + Q2) / 2.0
        #sigma_Q = torch.abs(Q1 - Q2) / 2.0
        sigma_Q = self.std(obs, action)
        if both_values:
            return mu_Q, sigma_Q
        if not upper_bound:
            return mu_Q

        Q_UB = mu_Q + beta_UB * sigma_Q
        return Q_UB

    def predict_ori(self, obs, action,  upper_bound=True, beta_UB=4.66, both_values=False, target=False):
        if isinstance(obs, np.ndarray):
            obs = torch_ify(obs)
            action = torch_ify(action)
        if len(obs.shape) == 1:
            args = list(torch.unsqueeze(i, dim=0) for i in (obs, action))
        else:
            args = list(i for i in (obs, action))
        if not target:
            Q1 = self.qfs[0](*args)
            Q2 = self.qfs[1](*args)
        else:
            Q1 = self.tfs[0](*args)
            Q2 = self.tfs[1](*args)
        mu_Q = (Q1 + Q2) / 2.0
        sigma_Q = torch.abs(Q1 - Q2) / 2.0
        if both_values:
            return mu_Q, sigma_Q
        if not upper_bound:
            return mu_Q

        Q_UB = mu_Q + beta_UB * sigma_Q
        return Q_UB


    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi +
                            self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 0

        """
        QF Loss
        """
        # q_preds = []
        # for i in range(len(self.qfs)):
        #     q_preds.append(self.qfs[i](obs, actions))
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic,
        )
        # target_qs = [q(next_obs, new_next_actions) for q in self.tfs]
        # target_qs = torch.stack(target_qs, dim=0)
        # target_q_values = torch.min(target_qs, dim=0)[0] - alpha * new_log_pi
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi
        q_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q_values
        # qf_losses = []
        # qf_loss = 0
        # for i in range(len(self.qfs)):
        #     q_loss = self.qf_criterion(q_preds[i], q_target.detach())
        #     qf_losses.append(q_loss)
        #     qf_loss += q_loss
        #     self.qf_optimizers[i].zero_grad()
        #     q_loss.backward()
        #     self.qf_optimizers[i].step()
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        qf_loss = qf1_loss + qf2_loss

        """
        STD Loss
        """
        new_next_actions_det, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True, deterministic=True
        )
        std_preds = self.std(obs, actions)
        if self.use_target_std:
            target_stds = self.std_target(next_obs, new_next_actions_det) 
        else:
            target_stds = self.std(next_obs, new_next_actions_det) 

        std_target = (1. - terminals) * self.discount * target_stds

        if self.prv_std_qty > 0:
            qty = int(np.round(obs.shape[0] * self.prv_std_qty))
            # qty = qty if qty > 0 else 1 # never irrelevant
            f_obs = torch.FloatTensor(qty, obs.shape[1]).uniform_(-1, 1)
            f_actions = torch.FloatTensor(qty, actions.shape[1]).uniform_(-1, 1)
            f_std_preds = self.std(f_obs, f_actions)
            f_q_preds = torch.min(
                self.qf1(f_obs, f_actions),
                self.qf2(f_obs, f_actions),
            ) # for eval stats only
            f_std_target = self.prv_std(f_obs, f_actions)
            f_std_target = torch.clamp(f_std_target, 0, self.std_init)

        if self.std_soft_update:
            current_stds = std_preds.detach()
            next_stds = (1. - terminals) * self.discount * target_stds
            std_target = self.std_soft_update_prob * next_stds + (1 - self.std_soft_update_prob) * current_stds 

        if self.counts:
            counts = batch['counts']
            factor = torch.zeros_like(counts)
            factor[counts == 0] = 1
            std_target = std_target * factor + (1 - factor) * std_preds   

        std_target = torch.clamp(std_target, 0, self.std_init)
        std_loss = self.qf_criterion(std_preds, std_target.detach())
        if self.prv_std_qty > 0:
            std_loss += self.prv_std_weight * self.qf_criterion(f_std_preds, f_std_target.detach())    

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.qf2_optimizer.step()

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.std_optimizer.zero_grad()
        std_loss.backward(retain_graph=True)
        self.std_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            # for i in range(len(self.qfs)):
            #     ptu.soft_update_from_to(
            #     self.qfs[i], self.tfs[i], self.soft_target_tau
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.std, self.std_target, self.soft_target_tau
            )


        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()
            # for i in range(len(self.qfs)):
            #     self.eval_statistics['QF' + str(i) + ' Loss'] = np.mean(ptu.get_numpy(qf_losses[i]))
            #     self.eval_statistics.update(create_stats_ordered_dict(
            #         'Q' + str(i) + 'Predictions',
            #         ptu.get_numpy(q_preds[i]),
            #     ))
            self.eval_statistics['QF mean'] = np.mean(ptu.get_numpy(torch.stack([q1_pred, q2_pred], dim=0))
                                                      , axis=0).mean()
            self.eval_statistics['QF std'] = np.mean(ptu.get_numpy(std_preds))
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Q Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Policy Grad'] = np.mean(ptu.get_numpy(grad_norm(self.policy)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Target',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics['STD Loss'] = np.mean(ptu.get_numpy(std_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q STD Predictions',
                ptu.get_numpy(std_preds),
            ))
            if self.prv_std_qty > 0:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q FSTD Predictions',
                    ptu.get_numpy(f_std_preds),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q F-Mean Predictions',
                    ptu.get_numpy(f_q_preds),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q FSTD Target',
                    ptu.get_numpy(f_std_target),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q STD Target',
                ptu.get_numpy(std_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            policy_mean.requires_grad_()
            a_grad = torch.autograd.grad(q_new_actions, policy_mean, grad_outputs=torch.ones_like(q_new_actions), create_graph=True)[0]
            gn = torch.norm(a_grad, dim=1)

            self.eval_statistics['Critic Action Grad'] = np.mean(ptu.get_numpy(gn))
            self.eval_statistics['Critic Target Action Grad'] = np.mean(ptu.get_numpy(gn))

            self.eval_statistics['Policy Grad'] = np.mean(ptu.get_numpy(grad_norm(self.policy)))

            self.eval_statistics['Target Policy Grad'] = np.mean(ptu.get_numpy(grad_norm(self.policy)))

        self._n_train_steps_total += 1

    def save_prv_std(self):
        if self.prv_std_qty > 0:
            if self.use_target_std:
                ptu.copy_model_params_from_to(self.std_target, self.prv_std)
            else: 
                ptu.copy_model_params_from_to(self.std, self.prv_std)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self) -> Iterable[nn.Module]:
        # return [self.policy] + self.qfs + self.tfs
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        snapshot = dict(
            policy_state_dict=self.policy.state_dict(),
            policy_optim_state_dict=self.policy_optimizer.state_dict(),
            qf1_state_dict=self.qf1.state_dict(),
            qf1_optim_state_dict=self.qf1_optimizer.state_dict(),
            target_qf1_state_dict=self.target_qf1.state_dict(),
            qf2_state_dict=self.qf2.state_dict(),
            qf2_optim_state_dict=self.qf2_optimizer.state_dict(),
            target_qf2_state_dict=self.target_qf2.state_dict(),
            eval_statistics=self.eval_statistics,
            _n_train_steps_total=self._n_train_steps_total,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
            )

        if self.use_automatic_entropy_tuning:
            snapshot['log_alpha'] = self.log_alpha
            snapshot['alpha_optim_state_dict'] = self.alpha_optimizer.state_dict()
        return snapshot

        # qfs_state_dicts = []
        # qfs_optims_state_dicts = []
        # target_qfs_state_dicts = []
        # for i in range(len(self.qfs)):
        #     qfs_state_dicts.append(self.qfs[i].state_dict())
        #     qfs_optims_state_dicts.append(self.qf_optimizers[i].state_dict())
        #     target_qfs_state_dicts.append(self.tfs[i].state_dict())
        #
        # data["qfs_state_dicts"] = qfs_state_dicts
        # data["qfs_optims_state_dicts"] = qfs_optims_state_dicts
        # data["target_qfs_state_dicts"] = target_qfs_state_dicts
        # return data


    def restore_from_snapshot(self, ss):
        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)

        # self.qfs_optimizer = []
        # self.qfs = []
        # self.tfs = []
        # qfs_state_dicts, qfs_optims_state_dicts = ss['qfs_state_dicts'], ss['qfs_optims_state_dicts']
        # target_qfs_state_dicts = ss['target_qfs_state_dicts']
        # for i in range(len(qfs_state_dicts)):
        #
        #     self.qfs[1].load_state_dict(qfs_state_dicts[i])
        #     self.qfs_optimizer[i].load_state_dict(qfs_optims_state_dicts[i])
        #     self.tfs[i].load_state_dict(target_qfs_state_dicts[i])
        qf1_state_dict, qf1_optim_state_dict = ss['qf1_state_dict'], ss['qf1_optim_state_dict']
        target_qf1_state_dict = ss['target_qf1_state_dict']

        self.qf1.load_state_dict(qf1_state_dict)
        self.qf1_optimizer.load_state_dict(qf1_optim_state_dict)
        self.target_qf1.load_state_dict(target_qf1_state_dict)

        qf2_state_dict, qf2_optim_state_dict = ss['qf2_state_dict'], ss['qf2_optim_state_dict']
        target_qf2_state_dict = ss['target_qf2_state_dict']

        self.qf2.load_state_dict(qf2_state_dict)
        self.qf2_optimizer.load_state_dict(qf2_optim_state_dict)
        self.target_qf2.load_state_dict(target_qf2_state_dict)
        if self.use_automatic_entropy_tuning:
            log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']
            self.log_alpha.data.copy_(log_alpha)
            self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)

        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
