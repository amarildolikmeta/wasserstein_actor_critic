import json
import sys
sys.path.append("../")
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.trainer import SACTrainer
import numpy as np
import torch
from utils.variant_util import env_producer, get_policy_producer, get_q_producer
from utils.pythonplusplus import load_gzip_pickle


ts = '1584884279.5007188'
ts = '1589352957.4422379'
iter = 190
path = '../data/point/sac_/' + ts
ts = '1590677750.0582957'
path = '../data/point/mean_update_counts/p-oac_/' + ts
ts = '1595343877.9346888'
path = '../data/point/hard/terminal/ddpgcounts/p-oac_/no_bias/' + ts

restore = True

variant = json.load(open(path + '/variant.json', 'r'))
domain = variant['domain']
seed = variant['seed']
r_max = variant['r_max']
ensemble = variant['ensemble']
delta = variant['delta']
n_estimators = variant['n_estimators']
if seed == 0:
    np.random.seed()
    seed = np.random.randint(0, 1000000)
torch.manual_seed(seed)
np.random.seed(seed)
env_args = {}
if domain in ['riverswim']:
    env_args['dim'] = variant['dim']
if domain in ['point']:
    env_args['difficulty'] = variant['difficulty']
    env_args['max_state'] = variant['max_state']
    env_args['clip_state'] = variant['clip_state']
    env_args['terminal'] = variant['terminal']

expl_env = env_producer(domain, seed, **env_args)
eval_env = env_producer(domain, seed * 10 + 1, **env_args)
obs_dim = expl_env.observation_space.low.size
action_dim = expl_env.action_space.low.size

# Get producer function for policy and value functions
M = variant['layer_size']
N = variant['num_layers']

alg = variant['alg']

if alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac'] and  variant['share_layers']:
    output_size = n_estimators
    n_estimators = 1
else:
    output_size = 1
ob = expl_env.reset()
print(ob)
q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
policy_producer = get_policy_producer(
    obs_dim, action_dim, hidden_sizes=[M] * N)
q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
alg_to_trainer = {
    'sac': SACTrainer,
    'oac': SACTrainer,
    'p-oac': ParticleTrainer,
    'g-oac': GaussianTrainer
}
trainer = alg_to_trainer[variant['alg']]

kwargs ={ }
if alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']:
    n_estimators = variant['n_estimators']
    kwargs = dict(
        n_estimators=n_estimators,
        delta=variant['delta'],
        q_min=q_min,
        q_max=q_max,
        ensemble=variant['ensemble'],
        n_policies=variant['n_policies'],
    )
kwargs.update(dict(
            policy_producer=policy_producer,
            q_producer=q_producer,
            action_space=expl_env.action_space,
        ))
print(kwargs)
kwargs.update(variant['trainer_kwargs'])
trainer = trainer(**kwargs)
# try:
#     experiment = path + '/best.zip_pkl'
#     exp = load_gzip_pickle(experiment)
#     print(exp['epoch'])
#     trainer.restore_from_snapshot(exp['trainer'])
# except:
experiment = path + '/params.zip_pkl'
exp = load_gzip_pickle(experiment)
print(exp['epoch'])
trainer.restore_from_snapshot(exp['trainer'])

for i in range(10):
    s = expl_env.reset()
    done = False
    ret = 0
    t = 0
    while not done and t < 400:
        expl_env.render()
        if hasattr(trainer, 'target_policy'):
            a, agent_info = trainer.target_policy.get_action(s, deterministic=True)
        else:
            a, agent_info = trainer.policy.get_action(s, deterministic=True)
        s, r, done, _ = expl_env.step(a)
        t += 1
        ret += r
    expl_env.render()
    print("Return: ", ret)
    input()