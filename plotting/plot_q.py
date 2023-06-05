import json
import sys

sys.path.append('../')
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
import numpy as np
import torch
from utils.variant_util import env_producer, get_policy_producer, get_q_producer
from utils.core import np_ify, torch_ify
import matplotlib.pyplot as plt
from utils.pythonplusplus import load_gzip_pickle
import argparse
import pickle
from matplotlib import gridspec

LEGEND_FONT_SIZE = 28
AXIS_FONT_SIZE = 18
TICKS_FONT_SIZE = 14
MARKER_SIZE = 10
LINE_WIDTH = 3.0
TITLE_SIZE = 22
FIG_WIDTH = 40
FIG_HEIGHT = 30
parser = argparse.ArgumentParser()
#parser.add_argument('--path', type=str, default='../data/mean_update_counts/riverswim/p-oac_/1584884279.5007188/')
#parser.add_argument('--path', type=str, default='./data/riverswim/25/global/p-oac_/p-oac/')
parser.add_argument('--path', type=str, default='./data/riverswim/25/002b/02-up/')
parser.add_argument('--iter', type=int, default=70)
parser.add_argument('--max_iter', type=int, default=71)
parser.add_argument('--delta_iter', type=int, default=10)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

base_dir = args.path
iter = 190
restore = True
path = base_dir + 'variant.json'
variant = json.load(open(path, 'r'))
domain = variant['domain']
seed = variant['seed']
r_max = variant['r_max']
ensemble = variant['ensemble']
delta = variant['delta']
alg = variant['alg']
n_estimators = variant['n_estimators']
mean_update = variant['trainer_kwargs']['mean_update']

if seed == 0:
    np.random.seed()
    seed = np.random.randint(0, 1000000)
torch.manual_seed(seed)
np.random.seed(seed)
env_args = {}
if domain in ['riverswim']:
    env_args['dim'] = variant['dim']
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
else:
    output_size = 1
ob = expl_env.reset()
print(ob)
q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
policy_producer = get_policy_producer(
    obs_dim, action_dim, hidden_sizes=[M] * N)
q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
if alg == 'p-oac':
    trainer_producer = ParticleTrainer
else:
    trainer_producer = GaussianTrainer
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
iter = args.iter
delta_iter = args.delta_iter
max_iter = args.max_iter
num_trains = variant['algorithm_kwargs']['num_trains_per_train_loop']
save_sampled_data = variant['algorithm_kwargs']['save_sampled_data'] and False
if save_sampled_data:
    all_sampled_states = pickle.load(open(base_dir + '/sampled_states.pkl', 'rb'))
    all_sampled_actions = pickle.load(open(base_dir + '/sampled_actions.pkl', 'rb'))
while iter <= max_iter:
    if restore:
        try:
            experiment = base_dir + '/itr_' + str(iter) + '.zip_pkl'
            exp = load_gzip_pickle(experiment)
            trainer.restore_from_snapshot(exp['trainer'])
        except:
            break

    delta_action = 0.05
    xs = np.array([-1 + i * delta_action for i in range(int(2 / delta_action + 1))])

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    arrangement = (5, 5, 5, 5, 5)
    if save_sampled_data:
        sampled_states = all_sampled_states[iter * num_trains: (iter + 1) * num_trains]
        sampled_actions = all_sampled_actions[iter * num_trains: (iter + 1) * num_trains]
        arrangement += (2,)
    nrows = len(arrangement)

    gs = gridspec.GridSpec(nrows, 1)
    ax_specs = []
    for r, ncols in enumerate(arrangement):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, ncols, subplot_spec=gs[r])
        for col in range(ncols):
            ax = plt.Subplot(fig, gs_row[col])
            fig.add_subplot(ax)

    # for i, ax in enumerate(fig.axes):
    #     ax.text(0.5, 0.5, "Axis: {}".format(i), fontweight='bold',
    #             va="center", ha="center")
    #     ax.tick_params(axis='both', bottom='off', top='off', left='off',
    #                    right='off', labelbottom='off', labelleft='off')

    # fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))
    ax = fig.axes
    max_state = eval_env.observation_space.high[0]
    for i in range(25):
        row = i // 5
        col = i % 5
        ob = i * max_state / 25
        ob = np.array([ob]).reshape((1, 1))
        action = eval_env.action_space.low[0]
        particles = []
        # axis = ax[row, col]
        axis = ax[i]
        if alg == 'p-oac':
            delta_index = trainer.delta_index
            while action < eval_env.action_space.high + delta_action:
                a = np.array([action]).reshape((1, 1))
                qs, _ = trainer.predict(ob, a, all_particles=True)
                qs = np_ify(qs)
                qs = qs.reshape((1, -1))
                qs = np.sort(qs, axis=1)
                particles.append(qs)
                action += delta_action
            particles = np.concatenate(particles, axis=0)
            mean_q = particles.mean(axis=1)
            std_q = particles.std(axis=1)
            xs = xs[:particles.shape[0]]
            axis.plot(xs, mean_q)
            axis.fill_between(xs, mean_q - 2 * std_q, mean_q + 2 * std_q, alpha=0.3)
            axis.plot(xs, particles[:, delta_index])
            # for i in range(particles.shape[1]):
            #     # axis.plot(xs, mean_q)
            #     axis.plot(xs, particles[:, i])
            #     # axis.fill_between(xs, mean_q - 2*std_q, mean_q+2*std_q, alpha=0.3)
            #     # axis.plot(xs, particles[:, delta_index])
            max_action = xs[np.argmax(particles[:, delta_index])]
            max_action_mean = xs[np.argmax(particles.mean(axis=1))]
            max_action_Q = particles[np.argmax(particles[:, delta_index]), delta_index]
            max_action_mean_Q = mean_q[np.argmax(particles.mean(axis=1))]
        else:
            qfs_array = []
            stds_array = []
            bounds_array = []
            while action < eval_env.action_space.high[0] + delta_action:
                a = np.array([action]).reshape((1, 1))
                qs, upper_bound = trainer.predict(ob, a, std=True)
                stds = qs[1]
                qs = qs[0]
                qs = np_ify(qs)
                stds = np_ify(stds)
                upper_bound = np_ify(upper_bound)
                qfs_array.append(qs)
                stds_array.append(stds)
                bounds_array.append(upper_bound)
                action += delta_action
            qfs_array = np.concatenate(qfs_array, axis=0)
            stds_array = np.concatenate(stds_array, axis=0)
            bounds_array = np.concatenate(bounds_array, axis=0).flatten()
            mean_q = qfs_array.flatten()
            std_q = stds_array.flatten()
            xs = xs[:qfs_array.shape[0]]
            axis.plot(xs, mean_q)
            axis.fill_between(xs, mean_q - 2 * std_q, mean_q + 2 * std_q, alpha=0.3)
            axis.plot(xs, bounds_array)
            max_action = xs[np.argmax(bounds_array)]
            max_action_mean = xs[np.argmax(mean_q)]
            max_action_Q = bounds_array[np.argmax(bounds_array)]
            max_action_mean_Q = mean_q[np.argmax(mean_q)]
        if ensemble:
            for p in range(len(trainer.policy.policies)):
                new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = trainer.policy.policies[p](
                    obs=torch_ify(ob), reparameterize=True, return_log_prob=True, deterministic=trainer.deterministic
                )
                axis.axvline(x=np_ify(new_obs_actions), c='red', label='opt_policy')
        else:
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = trainer.policy(
                obs=torch_ify(ob), reparameterize=True, return_log_prob=True, deterministic=trainer.deterministic
            )
            axis.axvline(x=np_ify(new_obs_actions), c='red', label='opt_policy' if i == 0 else None)
        if mean_update:
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = trainer.target_policy(
                obs=torch_ify(ob), reparameterize=True, return_log_prob=True, deterministic=trainer.deterministic)
            axis.axvline(x=np_ify(new_obs_actions), c='green', label='Mean policy' if i == 0 else None)
        # axis.axvline(x=max_action, c='blue', label='max_upper_bound' if i == 0 else None)
        # axis.axvline(x=max_action_mean, c='orange', label='max_mean' if i == 0 else None)
        axis.scatter(x=max_action, y=max_action_Q, c='blue', marker='x')
        axis.scatter(x=max_action_mean, y=max_action_mean_Q, c='orange', marker='x')
        # for k in range(particles.shape[-1]):
        #     xs = xs[:particles.shape[0]]
        #     axis.plot(xs, particles[:, k])
        #     if k == delta_index:
        #         max_action = xs[np.argmax(particles[:, k])]
        #         axis.axvline(x=max_action)
        axis.set_title("state-" + str(i), fontdict={'fontsize': TITLE_SIZE})
        axis.tick_params(labelsize=TICKS_FONT_SIZE)
        # axis.set_ylim((q_min, q_max))
    if save_sampled_data:
        states = np.concatenate(sampled_states)
        actions = np.concatenate(sampled_actions)
        ax[-2].hist(states)
        ax[-1].hist(actions)
        ax[-2].set_title("sampled states", fontdict={'fontsize': TITLE_SIZE})
        ax[-1].set_title("sampled actions", fontdict={'fontsize': TITLE_SIZE})
        ax[-2].set_xlabel("sampled states", fontdict={'fontsize': AXIS_FONT_SIZE})
        ax[-1].set_ylabel("sampled actions", fontdict={'fontsize': AXIS_FONT_SIZE})
        ax[-2].tick_params(labelsize=TICKS_FONT_SIZE)
        ax[-1].tick_params(labelsize=TICKS_FONT_SIZE)

    fig.legend()
    if args.show:
        plt.show()
    fig.savefig(base_dir + '/p_and_v_' + str(iter) + '.png')
    plt.close(fig)
    iter += delta_iter
