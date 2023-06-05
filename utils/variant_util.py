
import torch
import numpy as np
import random 

from replay_buffer import ReplayBuffer, ReplayBufferCount
from replay_buffer_no_resampling import ReplayBufferNoResampling
from utils.env_utils import NormalizedBoxEnv, env_producer
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from trainer.policies import TanhGaussianPolicy, MakeDeterministic, EnsemblePolicy, TanhGaussianMixturePolicy, GaussianTanhPolicy
from trainer.policies import GaussianPolicy
from trainer.trainer import SACTrainer
from trainer.ddpg_trainer import DDPGTrainer
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.gaussian_trainer_ts import GaussianTrainerTS
from trainer.particle_trainer_ts import ParticleTrainerTS
from trainer.gaussian_trainer_oac import GaussianTrainerOAC
from trainer.gaussian_trainer_soft import GaussianTrainerSoft
from trainer.particle_trainer_oac import ParticleTrainer as ParticleTrainerOAC
from networks import FlattenMlp

from torch.nn import functional as F

def get_policy_producer(
    obs_dim, action_dim,
    hidden_sizes,
    clip=True,
    std=None,
    policy_activation='ReLU',
    policy_output='TanhGaussian'
):

    if policy_activation == 'ReLU':
        hidden_activation = F.relu
    elif policy_activation == 'LeakyReLU':
        hidden_activation = F.leaky_relu
    else:
        raise ValueError('Activation not implemented')

    def policy_producer(
        deterministic=False,
        bias=None,
        ensemble=False,
        n_policies=1,
        n_components=1,
        approximator=None,
        share_layers=False
    ):
        if policy_output == 'GaussianTanh':
            policy = GaussianTanhPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_activation=hidden_activation,
                    hidden_sizes=hidden_sizes,
                    bias=bias,
                    std=std
            )
        elif ensemble:
            policy = EnsemblePolicy(approximator=approximator,
                                    hidden_sizes=hidden_sizes,
                                    obs_dim=obs_dim,
                                    action_dim=action_dim,
                                    n_policies=n_policies,
                                    bias=bias,
                                    share_layers=share_layers)
        else:
            if not clip:
                policy = GaussianPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    bias=bias
                )
            elif n_components == 1:
                policy = TanhGaussianPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_activation=hidden_activation,
                    hidden_sizes=hidden_sizes,
                    bias=bias,
                    std=std
                )

            else:
                policy = TanhGaussianMixturePolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    bias=bias,
                    n_components=n_components
                )
            '''
            policy = TanhGaussianMixturePolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                bias=bias,
                n_components=n_components
            )
            '''
            if deterministic:
                policy = MakeDeterministic(policy)

        return policy

    return policy_producer

def get_q_producer(obs_dim, action_dim, hidden_sizes, output_size=1):
    def q_producer(bias=None, positive=False, train_bias=True):
        return FlattenMlp(input_size=obs_dim + action_dim,
                            output_size=output_size,
                            hidden_sizes=hidden_sizes,
                            bias=bias,
                            positive=positive,
                            train_bias=train_bias)

    return q_producer

def build_variant(variant, return_replay_buffer=True, return_collectors=True):
    res = {}

    domain = variant['domain']
    seed = variant['seed']
    env_args = {}
    if domain in ['riverswim']:
        env_args['dim'] = variant['dim']
        env_args['deterministic'] = False
        if 'deterministic_rs' in variant:
            env_args['deterministic'] = variant['deterministic_rs']
    if domain in ['lqg']:
        env_args['sigma_noise'] = variant['sigma_noise']
    if domain in ['point']:
        env_args['difficulty'] = variant['difficulty']
        env_args['clip_state'] = variant['clip_state']
        env_args['terminal'] = variant['terminal']
        env_args['sparse_reward'] = variant['sparse_reward']
        env_args['max_state'] = variant['max_state']
    if domain in ['ant_maze']:
        env_args['difficulty'] = variant['difficulty']
        env_args['clip_state'] = variant['clip_state']
        env_args['terminal'] = variant['terminal']
    if 'cliff' in domain:
        env_args['sigma_noise'] = variant['sigma_noise']

    expl_env = env_producer(domain, seed, **env_args)
    eval_env = env_producer(domain, seed * 10 + 1, **env_args)
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    # Get producer function for policy and value functions
    M = variant['layer_size']
    N = variant['num_layers']
    n_estimators = variant['n_estimators']

    if variant['share_layers']:
        output_size = n_estimators
        # n_estimators = 1
    else:
        output_size = 1
    q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
    std = None
    if variant['algorithm_kwargs']['ddpg_noisy']:
        std = variant['std']
        std = np.ones(action_dim) * std
    
    policy_producer = get_policy_producer(
        obs_dim, action_dim, 
        hidden_sizes=[M] * N, 
        clip=variant['clip_action'], 
        std=std, 
        policy_activation=variant['policy_activation'], 
        policy_output=variant['policy_output']
    )
    # Finished getting producer

    if return_collectors:
        remote_eval_path_collector = MdpPathCollector(
            eval_env
        )

        # remote_eval_path_collector = RemoteMdpPathCollector.remote(
        #     domain, seed * 10 + 1,
        #     policy_producer
        # )

        expl_path_collector = MdpPathCollector(
            expl_env,
        )
    else:
        remote_eval_path_collector = None
        expl_path_collector = None

    if return_replay_buffer:
        if variant['alg'] in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac', 'oac-w', 'gs-oac'] and variant['trainer_kwargs']["counts"]:
            replay_buffer = ReplayBufferCount(
                variant['replay_buffer_size'],
                ob_space=expl_env.observation_space,
                action_space=expl_env.action_space,
                priority_sample=variant['priority_sample']
            )
        elif variant["no_resampling"] and variant['alg'] in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac', 'oac-w', 'gs-oac']:
            replay_buffer = ReplayBufferNoResampling(
                variant['replay_buffer_size'],
                ob_space=expl_env.observation_space,
                action_space=expl_env.action_space
            )
        else:
            replay_buffer = ReplayBuffer(
                variant['replay_buffer_size'],
                ob_space=expl_env.observation_space,
                action_space=expl_env.action_space
            )
    else:
        replay_buffer = None

    if variant['alg'] in ['ddpg']:
        trainer = DDPGTrainer(
            policy_producer,
            q_producer,
            action_space=expl_env.action_space,
            **variant['trainer_kwargs']
        )
    elif variant['alg'] in ['oac', 'sac']:
        trainer = SACTrainer(
            policy_producer,
            q_producer,
            action_space=expl_env.action_space,
            **variant['trainer_kwargs']
        )
    elif variant['alg'] == 'p-oac':
        if variant['optimistic_exp']['should_use']:
            trainer_class = ParticleTrainerOAC
            variant['trainer_kwargs']['deterministic'] = False
        else:
            trainer_class = ParticleTrainer
            variant['trainer_kwargs']['deterministic'] = not variant['stochastic']
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        trainer = trainer_class(
            policy_producer,
            q_producer,
            n_estimators=n_estimators,
            delta=variant['delta'],
            q_min=q_min,
            q_max=q_max,
            action_space=expl_env.action_space,
            ensemble=variant['ensemble'],
            n_policies=variant['n_policies'],
            **variant['trainer_kwargs']
        )
    elif variant['alg'] in ['g-oac', 'oac-w', 'gs-oac']:
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        if variant['alg'] == 'g-oac': 
            trainer = GaussianTrainer(
                policy_producer,
                q_producer,
                n_estimators=n_estimators,
                delta=variant['delta'],
                q_min=q_min,
                q_max=q_max,
                action_space=expl_env.action_space,
                pac=variant['pac'],
                ensemble=variant['ensemble'],
                n_policies=variant['n_policies'],
                **variant['trainer_kwargs']
            )
        elif variant['alg'] == 'oac-w':
            trainer = GaussianTrainerOAC(
                policy_producer,
                q_producer,
                q_min=q_min,
                q_max=q_max,
                action_space=expl_env.action_space,
                **variant['trainer_kwargs']
            )
        elif variant['alg'] == 'gs-oac':
            trainer = GaussianTrainerSoft(
                policy_producer,
                q_producer,
                delta=variant['delta'],
                q_min=q_min,
                q_max=q_max,
                action_space=expl_env.action_space,
                **variant['trainer_kwargs']
            )
    elif variant['alg'] == 'g-tsac':
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        q_posterior_producer = None
        if variant['share_layers']:
            q_posterior_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=1)
        trainer = GaussianTrainerTS(
            policy_producer,
            q_producer,
            n_estimators=n_estimators,
            delta=variant['delta'],
            q_min=q_min,
            q_max=q_max,
            action_space=expl_env.action_space,
            n_components=variant['n_components'],
            q_posterior_producer=q_posterior_producer,
            **variant['trainer_kwargs']
        )
    elif variant['alg'] == 'p-tsac':
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        q_posterior_producer = None
        if variant['share_layers']:
            q_posterior_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=1)
        trainer = ParticleTrainerTS(
            policy_producer,
            q_producer,
            n_estimators=n_estimators,
            delta=variant['delta'],
            q_min=q_min,
            q_max=q_max,
            action_space=expl_env.action_space,
            n_components=variant['n_components'],
            q_posterior_producer=q_posterior_producer,
            **variant['trainer_kwargs']
        )
    else:
        raise ValueError("Algorithm no implemented:" + variant['alg'])

    res['trainer'] = trainer
    res['expl_path_collector'] = expl_path_collector
    res['remote_eval_path_collector'] = remote_eval_path_collector
    res['replay_buffer'] = replay_buffer

    return res 


