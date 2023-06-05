import numpy as np
import torch.optim as optim
import torch
from torch import nn as nn
from trainer.trainer import SACTrainer
import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
from scipy.stats import norm
from utils.core import torch_ify, optimize_policy, grad_norm
from utils.misc import mellow_max


class GaussianTrainerSoft(SACTrainer):
    def __init__(
            self,
            policy_producer,
            q_producer,
            action_space=None,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=3e-4,
            target_policy_lr=None,
            qf_lr=3e-4,
            std_lr=3e-5,
            optimizer_class=optim.Adam,
            policy_weight_decay=0, # FIXME: remove it 
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            deterministic=False,
            fixed_alpha=0,
            delta=0.95,
            q_min=0,
            q_max=100,
            counts=False,
            mean_update=False,
            std_soft_update=False,
            std_soft_update_prob=0., # ??
            train_bias=True,
            use_target_policy=False,
            prv_std_qty=0,
            prv_std_weight=1,
            dont_use_target_std=False,
            policy_grad_steps=1,
            stable_critic=False
    ):
        super().__init__(policy_producer,
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
                         deterministic=deterministic)
        """
        The class mutable state
        """
        self.stable_critic = stable_critic
        self.fixed_alpha = fixed_alpha

        self.action_space = action_space
        self.q_min = q_min
        self.q_max = q_max
        self.standard_bound = standard_bound = norm.ppf(delta, loc=0, scale=1)
        mean = (q_max + q_min) / 2 
        # self.qf1 = q_producer(bias=mean)
        # self.qf2 = q_producer(bias=mean)
        # self.target_qf1 = q_producer(bias=mean) 
        # self.target_qf2 = q_producer(bias=mean)
        std = (q_max - q_min) / np.sqrt(12)
        log_std = np.log(std)
        self.delta = delta
        self.std_init = std
        self.qfs = []
        self.qf_optimizers = []
        self.tfs = []
        self.mellow_max = mellow_max
        self.counts = counts
        self.mean_update = mean_update
        self.prv_std_qty = prv_std_qty
        self.prv_std_weight = prv_std_weight
        self.std_soft_update = std_soft_update
        self.std_soft_update_prob = std_soft_update_prob
        self.use_target_std = not dont_use_target_std
        self.policy_grad_steps = policy_grad_steps

        assert not self.counts or not self.std_soft_update

        self.std1 = q_producer(bias=log_std, positive=True, train_bias=train_bias)
        self.std1_optimizer = optimizer_class(
            self.std1.parameters(),
            lr=std_lr)
        if self.use_target_std:
            self.target_std1 = q_producer(bias=log_std, positive=True, train_bias=train_bias)
        if self.stable_critic:
            self.std2 = q_producer(bias=log_std, positive=True, train_bias=train_bias)
            self.std2_optimizer = optimizer_class(self.std2.parameters(), lr=std_lr)
            if self.use_target_std:
                self.target_std2 = q_producer(bias=log_std, positive=True, train_bias=train_bias)

        # self.qfs = [self.q, self.std]
        # self.tfs = [self.q_target, self.target_std]
        if self.prv_std_qty > 0:
            self.prv_std1 = q_producer(bias=log_std, positive=True) 
            # ptu.copy_model_params_from_to(self.std1, self.prv_std1)
            if self.stable_critic: 
                self.prv_std2 = q_producer(bias=log_std, positive=True) 
                # ptu.copy_model_params_from_to(self.std2, self.prv_std2)
        self.target_policy = policy_producer()
        if target_policy_lr is None:
            target_policy_lr = policy_lr
        self.target_policy_optimizer = optimizer_class(
            self.target_policy.parameters(), 
            lr=target_policy_lr
        )

        # target_policy in ddpg
        self.use_target_policy = use_target_policy
        if use_target_policy and not self.mean_update:
            self.target_policy_network = policy_producer()
            ptu.soft_update_from_to(
                self.policy, self.target_policy_network, 1
            )
        
        if self.use_automatic_entropy_tuning:
            self.log_target_alpha = ptu.zeros(1, requires_grad=True)
            self.target_alpha_optimizer = optimizer_class(
                [self.log_target_alpha],
                lr=policy_lr
            )

    def predict(self, obs, action, std=True, target=False):
        if isinstance(obs, np.ndarray):
            obs = torch_ify(obs)
            action = torch_ify(action)
        if self.stable_critic:
            ub_idx = torch.stack((
                self.qf1(obs, action) + self.standard_bound * self.std1(obs, action),
                self.qf2(obs, action) + self.standard_bound * self.std2(obs, action)
            )).argmin(dim=0)
            qs = torch.where(ub_idx == 0, self.qf1(obs, action), self.qf2(obs, action))
            # qs = self.qf1(obs, action) if ub_idx == 0 else self.qf2(obs, action)
            stds = torch.where(ub_idx == 0, self.std1(obs, action), self.std2(obs, action))
            # stds = self.std1(obs, action) if ub_idx == 0 else self.std2(obs, action)
        else:
            qs = self.qf1(obs, action) + self.standard_bound * self.std1(obs, action)
            stds = self.std1(obs, action)
        upper_bound = qs + self.standard_bound * stds
        if std:
            return [qs, stds], upper_bound
        return upper_bound

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        discount = self.discount

        """
        Alpha Loss and Update
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        )
        new_target_obs_actions, target_policy_mean, target_policy_log_std, target_log_pi, *_ = self.target_policy(
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

            target_alpha_loss = -(self.log_target_alpha *
                           (target_log_pi +
                            self.target_entropy).detach()).mean()
            self.target_alpha_optimizer.zero_grad()
            target_alpha_loss.backward()
            self.target_alpha_optimizer.step()
            target_alpha = self.log_target_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.fixed_alpha
            target_alpha_loss = 0
            target_alpha = self.fixed_alpha
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        if self.stable_critic: 
            q2_pred = self.qf2(obs, actions)

        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        if self.mean_update:
            new_next_actions, _, _, new_log_pi, *_ = self.target_policy(
                obs=next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
            )
        else:
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                obs=next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
            )
        if self.stable_critic:
            t_qf1 = self.target_qf1(next_obs, new_next_actions)
            t_qf2 = self.target_qf2(next_obs, new_next_actions)

            if self.use_target_std:
                t_std1 = self.target_std1(next_obs, new_next_actions)
                t_std2 = self.target_std2(next_obs, new_next_actions)
            else:
                t_std1 = self.std1(next_obs, new_next_actions)
                t_std2 = self.std2(next_obs, new_next_actions)
        
            ub_idx = torch.stack((
                t_qf1 + self.standard_bound * t_std1,
                t_qf2 + self.standard_bound * t_std2
            )).argmin(dim=0)

            target_q_values = torch.where(ub_idx == 0, t_qf1, t_qf2) - target_alpha * new_log_pi
        else:
            t_qf1 = self.target_qf1(next_obs, new_next_actions)
            if self.use_target_std:
                t_std1 = self.target_std1(next_obs, new_next_actions)
            else:
                t_std1 = self.std1(next_obs, new_next_actions)
            target_q_values = t_qf1 - target_alpha * new_log_pi

        # q network loss
        q_target = self.reward_scale * rewards + \
                    (1. - terminals) * self.discount * target_q_values

        qf1_loss = qf_loss = self.qf_criterion(q1_pred, q_target.detach())
        if self.stable_critic:
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
            qf_loss = qf1_loss + qf2_loss
        
        """
        Update Q(s) Networks
        """

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        if self.stable_critic:
            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            self.qf2_optimizer.step()

        """
        Std Loss
        """
        std1_preds = std_preds = self.std1(obs, actions)
        if self.stable_critic:
            std2_preds = self.std2(obs, actions)
            std_preds = torch.where(ub_idx == 0, std1_preds, std2_preds)
            target_std = torch.where(ub_idx == 0, t_std1, t_std2)
        else:
            target_std = t_std1

        std_target = (1. - terminals) * discount * target_std

        if self.prv_std_qty > 0:
            qty = int(np.round(obs.shape[0] * self.prv_std_qty))
            # qty = qty if qty > 0 else 1 # never irrelevant
            f_obs = torch.FloatTensor(qty, obs.shape[1]).uniform_(-1, 1)
            f_actions = torch.FloatTensor(qty, actions.shape[1]).uniform_(-1, 1)
            f_std1_preds = self.std1(f_obs, f_actions)
            f_std1_target = self.prv_std1(f_obs, f_actions)
            f_std1_target = torch.clamp(f_std1_target, 0, self.std_init)

            if self.stable_critic:
                f_std2_preds = self.std2(f_obs, f_actions)
                f_std2_target = self.prv_std2(f_obs, f_actions)
                f_std2_target = torch.clamp(f_std2_target, 0, self.std_init)

                # < for eval stats
                f_std_preds = (f_std1_preds + f_std2_preds) / 2
                f_q_preds = (self.qf1(f_obs, f_actions) + self.qf2(f_obs, f_actions)) / 2
                f_std_target = (f_std1_target + f_std2_target) / 2
            else:
                f_std_preds = f_std1_preds
                f_q_preds = self.qf1(f_obs, f_actions)
                f_std_target = f_std1_target
                # for eval stats >

        if self.counts:
            counts = batch['counts']
            factor = torch.zeros_like(counts)
            factor[counts == 0] = 1
            std_target = std_target * factor + (1 - factor) * std_preds

        std_target = torch.clamp(std_target, 0, self.std_init)
        std1_loss = std_loss = self.qf_criterion(std1_preds, std_target.detach())
        if self.stable_critic:
            std2_loss = self.qf_criterion(std2_preds, std_target.detach())
            std_loss = std1_loss + std2_loss
        if self.prv_std_qty > 0:
            std1_loss += self.prv_std_weight * self.qf_criterion(f_std1_preds, f_std1_target.detach())
            if self.stable_critic:
                std2_loss += self.prv_std_weight * self.qf_criterion(f_std2_preds, f_std2_target.detach())
        """
        Update Std Networks
        """

        self.std1_optimizer.zero_grad()
        std1_loss.backward(retain_graph=True)
        self.std1_optimizer.step()

        if self.stable_critic:
            self.std2_optimizer.zero_grad()
            std2_loss.backward(retain_graph=True)
            self.std2_optimizer.step()

        """
        Update target policy
        """
        if self.stable_critic:
            t_q1_pred = self.qf1(obs, new_target_obs_actions)
            t_q2_pred = self.qf2(obs, new_target_obs_actions)
            q_new_target_actions = torch.min(
                t_q1_pred,
                t_q2_pred
            )
        else:
            q_new_target_actions = self.qf1(obs, new_target_obs_actions)

        target_policy_loss = (target_alpha * target_log_pi - q_new_target_actions).mean()
        self.target_policy_optimizer.zero_grad()
        target_policy_loss.backward(retain_graph=True)
        self.target_policy_optimizer.step()

        """
        Update policy
        """
        q1_new_actions = self.qf1(obs, new_obs_actions)
        std1_new_actions = self.std1(obs, new_obs_actions)

        if self.stable_critic:
            q2_new_actions = self.qf2(obs, new_obs_actions)
            std2_new_actions = self.std2(obs, new_obs_actions)
            upper_bound = torch.min(
                q1_new_actions + self.standard_bound * std1_new_actions,
                q2_new_actions + self.standard_bound * std2_new_actions
            )
        else:
            upper_bound = q1_new_actions + self.standard_bound * std1_new_actions
        policy_loss = (alpha * log_pi - upper_bound).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
            if self.use_target_std:
                ptu.soft_update_from_to(
                    self.std1, self.target_std1, self.soft_target_tau
                )
                if self.stable_critic:
                    ptu.soft_update_from_to(
                        self.std2, self.target_std2, self.soft_target_tau
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
            if self.stable_critic:
                q_pred = torch.stack((q1_pred, q2_pred)).mean(dim=0)
            else:
                q_pred = q1_pred

            self.eval_statistics['QF mean'] = np.mean(ptu.get_numpy(q_pred))
            self.eval_statistics['QF std'] = np.mean(ptu.get_numpy(std_preds))
            self.eval_statistics['Q Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
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
            target_policy_mean.requires_grad_()
            a_grad = torch.autograd.grad(
                q_new_target_actions, 
                target_policy_mean, 
                grad_outputs=torch.ones_like(q_new_target_actions), create_graph=True
            )[0]
            gn = torch.norm(a_grad, dim=1)

            self.eval_statistics['Critic Target Action Grad'] = np.mean(ptu.get_numpy(gn))

            new_target_obs_actions.requires_grad_()
            a_grad = torch.autograd.grad(
                q_new_target_actions,
                new_target_obs_actions,
                grad_outputs=torch.ones_like(q_new_target_actions), create_graph=True
            )[0]
            gn = torch.norm(a_grad, dim=1)

            self.eval_statistics['Critic Target Action Grad PTH'] = np.mean(ptu.get_numpy(gn))

            policy_loss = (upper_bound).mean()
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            target_policy_loss = (q_new_target_actions).mean()
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                target_policy_loss
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
                self.eval_statistics['Target Alpha'] = target_alpha.item()
                self.eval_statistics['Target Alpha Loss'] = target_alpha_loss.item()

            self.eval_statistics['Policy Grad'] = np.mean(ptu.get_numpy(grad_norm(self.policy)))
            self.eval_statistics['Target Policy Grad'] = np.mean(ptu.get_numpy(grad_norm(self.target_policy)))

            w_norm = 0 
            for l in self.target_policy.fcs:
                w_norm += torch.norm(l.weight.T)
            w_norm += torch.norm(self.target_policy.last_fc.weight.T)

            self.eval_statistics['weights norm'] = np.mean(ptu.get_numpy(w_norm))

            w_bias = 0 
            for l in self.target_policy.fcs:
                w_bias += torch.norm(l.bias.T)
            w_bias += torch.norm(self.target_policy.last_fc.bias.T)

            self.eval_statistics['weights bias'] = np.mean(ptu.get_numpy(w_bias))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
        self._n_train_steps_total += 1

    def save_prv_std(self):
        if self.prv_std_qty > 0:
            ptu.copy_model_params_from_to(self.std1, self.prv_std1)
            if self.stable_critic:
                ptu.copy_model_params_from_to(self.std2, self.prv_std2)
        
        # ptu.copy_model_params_from_to(self.init_policy, self.policy_2)
    
    # def random_std_increase(self, batch_size, obs_dim, ac_dim):
    #     obs = torch.FloatTensor(batch_size, obs_dim).uniform_(-1, 1) 
    #     actions = torch.FloatTensor(batch_size, ac_dim).uniform_(-1, 1)
    #     if self.share_layers:
    #         raise ValueError('Not implemented')
    #         q_preds = self.q(obs, actions)
    #         std_preds =  q_preds[:, 1].unsqueeze(-1)
    #         optimizer = self.q_optimizer
    #     else:
    #         std_preds = self.std(obs, actions)
    #         optimizer = self.std_optimizer
    #     if self.std_inc_init:
    #         std_target = torch.zeros([batch_size, 1], dtype=torch.float32) + self.std_init
    #     else: 
    #         std_target = std_preds * 1/self.discount
    #         std_target = torch.clamp(std_target, 0, self.std_init)
    #         std_target = std_target
    #     std_loss = self.qf_criterion(std_preds, std_target.detach())
    #     optimizer.zero_grad()
    #     std_loss.backward(retain_graph=True)
    #     optimizer.step()

    @property
    def networks(self) -> Iterable[nn.Module]:

        networks = [self.policy] + self.qfs + self.tfs
        # if self.mean_update:
        networks += [self.target_policy]

        return networks

    def get_snapshot(self):
        data = dict(
            policy_state_dict=self.policy.state_dict(),
            policy_optim_state_dict=self.policy_optimizer.state_dict(),
            #qf1_state_dict=self.qf1.state_dict(),
            #qf1_optim_state_dict=self.qf1_optimizer.state_dict(),
            #target_qf1_state_dict=self.target_qf1.state_dict(),
            #qf2_state_dict=self.qf2.state_dict(),
            #qf2_optim_state_dict=self.qf2_optimizer.state_dict(),
            #target_qf2_state_dict=self.target_qf2.state_dict(),

            eval_statistics=self.eval_statistics,
            _n_train_steps_total=self._n_train_steps_total,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
        )
        if self.use_automatic_entropy_tuning:
            data['log_alpha'] = self.log_alpha
            data['alpha_optim_state_dict'] = self.alpha_optimizer.state_dict()
            data['log_target_alpha'] = self.log_target_alpha
            data['target_alpha_optim_state_dict'] = self.target_alpha_optimizer.state_dict()

        qfs_state_dicts = []
        qfs_optims_state_dicts = []
        target_qfs_state_dicts = []

        qfs_state_dicts.append(self.qf1.state_dict())
        qfs_optims_state_dicts.append(self.qf1_optimizer.state_dict())
        target_qfs_state_dicts.append(self.target_qf1.state_dict())
        if self.stable_critic:
            qfs_state_dicts.append(self.qf2.state_dict())
            qfs_optims_state_dicts.append(self.qf2_optimizer.state_dict())
            target_qfs_state_dicts.append(self.target_qf2.state_dict())

        std_state_dicts = []
        std_optims_state_dicts = []
        target_std_state_dicts = []

        #qfs_state_dicts.append(self.q.state_dict())
        #qfs_optims_state_dicts.append(self.q_optimizer.state_dict())
        #target_qfs_state_dicts.append(self.q_target.state_dict())
        
        std_state_dicts.append(self.std1.state_dict())
        std_optims_state_dicts.append(self.std1_optimizer.state_dict())
        if self.use_target_std:
            target_std_state_dicts.append(self.target_std1.state_dict())

        if self.stable_critic:
            std_state_dicts.append(self.std2.state_dict())
            std_optims_state_dicts.append(self.std2_optimizer.state_dict())
            if self.use_target_std:
                target_std_state_dicts.append(self.target_std2.state_dict())
            
        if self.prv_std_qty > 0:
            prv_std_state_dicts = []
            prv_std_state_dicts.append(self.prv_std1.state_dict())
            if self.stable_critic:
                prv_std_state_dicts.append(self.prv_std2.state_dict())
            data["prv_std_state_dicts"] = prv_std_state_dicts

        data["qfs_state_dicts"] = qfs_state_dicts
        data["qfs_optims_state_dicts"] = qfs_optims_state_dicts
        data["target_qfs_state_dicts"] = target_qfs_state_dicts

        data["std_state_dicts"] = std_state_dicts
        data["std_optims_state_dicts"] = std_optims_state_dicts
        data["target_std_state_dicts"] = target_std_state_dicts

        # if self.mean_update:
        data["target_policy_state_dict"] = self.target_policy.state_dict()
        data["target_policy_opt_state_dict"] = self.target_policy_optimizer.state_dict()
        return data

    def restore_from_snapshot(self, ss):
        # policy
        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)
        self.qfs_optimizer = []

        # qs
        qfs_state_dicts, qfs_optims_state_dicts = ss['qfs_state_dicts'], ss['qfs_optims_state_dicts']
        target_qfs_state_dicts = ss['target_qfs_state_dicts']

        self.qf1.load_state_dict(qfs_state_dicts[0])
        self.qf1_optimizer.load_state_dict(qfs_optims_state_dicts[0])
        self.target_qf1.load_state_dict(target_qfs_state_dicts[0])
        if self.stable_critic:
            self.qf2.load_state_dict(qfs_state_dicts[1])
            self.qf2_optimizer.load_state_dict(qfs_optims_state_dicts[1])
            self.target_qf2.load_state_dict(target_qfs_state_dicts[1])

        # std
        std_state_dicts, std_optims_state_dicts = ss['std_state_dicts'], ss['std_optims_state_dicts']
        target_std_state_dicts = ss['target_std_state_dicts']

        self.std1.load_state_dict(std_state_dicts[0])
        self.std1_optimizer.load_state_dict(std_optims_state_dicts[0])
        if self.use_target_std:
            self.target_std1.load_state_dict(target_std_state_dicts[0])
        if self.stable_critic:
            self.std2.load_state_dict(std_state_dicts[1])
            self.std2_optimizer.load_state_dict(std_optims_state_dicts[1])
            if self.use_target_std:
                self.target_std2.load_state_dict(target_std_state_dicts[1])

        if self.prv_std_qty > 0 and "prv_std_state_dicts" in ss:
            prv_std_state_dicts = ss["prv_std_state_dicts"]
            self.prv_std1.load_state_dict(prv_std_state_dicts[0])
            if self.stable_critic:
                self.prv_std2.load_state_dict(prv_std_state_dicts[1])

        log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']
        log_target_alpha, target_alpha_optim_state_dict = ss['log_target_alpha'], ss['target_alpha_optim_state_dict']

        self.log_alpha.data.copy_(log_alpha)
        self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)
        self.log_target_alpha.data.copy_(log_target_alpha)
        self.target_alpha_optimizer.load_state_dict(target_alpha_optim_state_dict)

        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']

        # if self.mean_update:
        self.target_policy.load_state_dict(ss["target_policy_state_dict"])
        self.target_policy_optimizer.load_state_dict(ss["target_policy_opt_state_dict"])

        if self.use_target_policy:
            self.target_policy_network.load_state_dict(ss["target_policy_network"])

    def obj_func(self, states, actions, upper_bound=False):
        qs = self.q(states, actions)
        if self.share_layers:
            raise ValueError('Not implemented')
            stds = qs[:, 1].unsqueeze(-1)
            qs = qs[:, 0].unsqueeze(-1)
        else:
            stds = self.std(states, actions)
        if upper_bound:
            obj = qs + self.standard_bound * stds
        else:
            obj = qs
        return obj

    def optimize_policies(self, buffer, out_dir='', epoch=0, save_fig=False):
        if self.ensemble:
            return
        optimize_policy(self.policy, self.policy_optimizer, buffer, obj_func=self.obj_func,
                        init_policy=self.init_policy, action_space=self.action_space, upper_bound=True,
                        out_dir=out_dir, epoch=epoch, save_fig=save_fig)
