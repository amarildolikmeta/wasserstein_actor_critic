from trainer.particle_trainer import ParticleTrainer
import json
from utils.variant_util import env_producer, get_policy_producer, get_q_producer
from envs.river_swim_continuous import RiverSwimContinuous
from trainer.trainer import SACTrainer
from utils.core import np_ify, torch_ify
import numpy as np

#path = './data/point/hard/mean_update_counts/p-oac_/iters/'
#path = './data/riverswim/25/counts/p-oac_/iters/'
path = './data/riverswim/25/p-oac_/name/'
base_dir = path
experiment = path + 'itr_2.zip_pkl'
variant = json.load(open(path + 'variant.json', 'r'))

trainer_producer = ParticleTrainer
trainer_producer = ()
domain = variant['domain']
seed = variant['seed']
r_max = variant['r_max']
ensemble = variant['ensemble']
delta = variant['delta']
alg = variant['alg']
n_estimators = variant['n_estimators']
#mean_update = variant['trainer_kwargs']['mean_update']

env_args = {}

if domain in ['point']:
        env_args['difficulty'] = variant['difficulty']
        env_args['clip_state'] = variant['clip_state']
        env_args['terminal'] = variant['terminal']
        env_args['max_state'] = variant['max_state']

expl_env = env_producer(domain, seed, **env_args)
eval_env = env_producer(domain, seed * 10 + 1, **env_args)

obs_dim = expl_env.observation_space.low.size
action_dim = expl_env.action_space.low.size

M = variant['layer_size']
N = variant['num_layers']
n_estimators = variant['n_estimators']

if variant['share_layers']:
    output_size = n_estimators
else:
    output_size = 1
ob = expl_env.reset()

q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
policy_producer = get_policy_producer(
    obs_dim, action_dim, hidden_sizes=[M] * N)
q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
if alg == 'p-oac':
    trainer_producer = ParticleTrainer
    trainer = trainer_producer(
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
else:
    trainer = SACTrainer(
            policy_producer,
            q_producer,
            action_space=expl_env.action_space,
            **variant['trainer_kwargs']
        )




from utils.pythonplusplus import load_gzip_pickle

#experiment = base_dir + 'params.zip_pkl'
experiment = path + 'itr_40.zip_pkl'
exp = load_gzip_pickle(experiment)
trainer.restore_from_snapshot(exp['trainer'])

mdp = RiverSwimContinuous(dim=25)
s = mdp.reset()
rets = []
n = 100
for i in range(n):
    t = 0
    ret = 0
    s = mdp.reset()
    while t < 100:
        #print(s)
        a, *_ = trainer.policy(
            obs=torch_ify(s), reparameterize=True,
            return_log_prob=True, 
            deterministic=trainer.deterministic
        )
        a = np_ify(a)
        a = 1
        sp = s
        s, r, _, _ = mdp.step(a)
        print(sp, a, r)
        ret += r
        t += 1
    rets.append(ret)
print("Average Return:", np.mean(rets))
print("Average error:", np.std(rets) / np.sqrt(n))










