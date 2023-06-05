import json
import sys
import seaborn as sns
sys.path.append('../')
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.trainer import SACTrainer
import numpy as np
import torch
from utils.variant_util import env_producer, get_policy_producer, get_q_producer
from utils.core import np_ify, torch_ify
import matplotlib.pyplot as plt
from utils.pythonplusplus import load_gzip_pickle
import argparse
import pickle
import pandas as pd

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
#parser.add_argument('--path', type=str, default='./data/point/hard/mean_update_counts/p-oac_/samples/')
parser.add_argument('--path', type=str, default='./data/point/hard/mean_update_counts/p-oac_/iters/')
parser.add_argument('--iter', type=int, default=80)
parser.add_argument('--max_iter', type=int, default=400)
parser.add_argument('--delta_iter', type=int, default=10)
parser.add_argument('--show', action='store_true')
parser.add_argument('--point', action='store_true')
parser.add_argument('--density_map', default=False)

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
batch_size = variant['algorithm_kwargs']['batch_size']
alg = variant['alg']
n_estimators = variant['n_estimators']
#mean_update = variant['trainer_kwargs']['mean_update']

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
elif alg == 'g-oac':
    trainer_producer = GaussianTrainer
else:
    trainer_producer = SACTrainer
if alg in ['p-oac', 'g-oac']:
    variant['trainer_kwargs'].update(dict(
        n_estimators=n_estimators,
        delta=variant['delta'],
        q_min=q_min,
        q_max=q_max,
    ))
trainer = trainer_producer(
    policy_producer,
    q_producer,
    action_space=expl_env.action_space,
    **variant['trainer_kwargs']
)
iter = args.iter
delta_iter = args.delta_iter
max_iter = args.max_iter
do_density_map = args.density_map
do_heatmap = True
num_trains = variant['algorithm_kwargs']['num_trains_per_train_loop']
save_sampled_data = variant['algorithm_kwargs']['save_sampled_data']
if save_sampled_data:
    try:
        all_sampled_states = pickle.load(open(base_dir + '/sampled_states.pkl', 'rb'))

        sampled_states = np.concatenate(all_sampled_states)
    except:
        data = pd.read_csv(base_dir + '/sampled_states.csv', header=None)
        sampled_states = data.iloc[:, :2].values
   # all_sampled_actions = pickle.load(open(base_dir + '/sampled_actions.pkl', 'rb'))
else:
    print("Need sampled data! Exiting!")
    exit(1)

iter = 80
max_iter = 81 # only 1 heatmap
while iter <= max_iter:
    if do_heatmap:
        try:
            experiment = base_dir + '/itr_' + str(iter) + '.zip_pkl'
            #experiment = base_dir + 'params.zip_pkl' # ???
            exp = load_gzip_pickle(experiment)
            trainer.restore_from_snapshot(exp['trainer'])
            heatmap = True
        except:
            heatmap = False
            iter += delta_iter
            continue
    else:
        heatmap = False
    delta_action = 1#delta_action = 0.25
    delta_state = 1#delta_state = 0.25
    try:
        min_state, max_state = eval_env.bounds[0][0], eval_env.bounds[1][0]
        min_y, max_y = eval_env.bounds[0][1], eval_env.bounds[1][1]
    except :
        min_state, max_state = eval_env.observation_space.low[0], eval_env.observation_space.high[0]
        min_y, max_y = eval_env.observation_space.low[1], eval_env.observation_space.high[1]
        if save_sampled_data:
            min_y = np.min(sampled_states, axis=0)[1]
            max_y = np.max(sampled_states, axis=0)[1]
            max_v = np.max(np.abs([min_y, max_y]))
            min_y = - max_v
            max_y = max_v
            delta_action = 0.1
            delta_state = 0.1
    if heatmap:
        ys = np.array([min_y + i * delta_action for i in range(int((max_y - min_y) / delta_action + 1))])[::-1]
        xs = np.array([min_state + i * delta_state for i in range(int((max_state - min_state) / delta_action + 1))])

        var = (max_state - min_state) / delta_action + 1

        data_heatmap = np.zeros((len(xs), len(ys)))
        num_samples = 1 #num_samples = 10
        for i in range(len(xs)):
            for j in range(len(ys)):
                o = np.array([xs[i], ys[j], 0.])
                s = 0
                for k in range(num_samples):
                    action = eval_env.action_space.sample()
                    v = np.random.uniform(low=-5, high=5, size=2)
                    ob = np.concatenate([o, v, np.array([0.])])
                    ob = np.array([ob]).reshape((1, 6))
                    a = np.array(action).reshape((1, action.shape[-1]))
                    if alg in ['p-oac']: # compute variance from particles
                        qs, _ = trainer.predict(ob, a, all_particles=True)
                        qs = np_ify(qs)
                        qs = qs.reshape((1, -1))
                        qs = np.sort(qs, axis=1)
                        std = np.std(qs.flatten())
                    elif alg in ['g-oac']:
                        qs, upper_bound = trainer.predict(ob, a, std=True)
                        stds = qs[1]
                        stds = np_ify(stds)
                        std = stds[0]
                    else:
                        qs, stds = trainer.predict(ob, a, both_values=True)
                        stds = np_ify(stds)
                        std = stds[0]
                    s += std
                std = s / num_samples
                data_heatmap[i, j] = std
    if (iter + 1) * num_trains * batch_size > sampled_states.shape[0]:
        break
    states = sampled_states[: (iter + 1) * num_trains * batch_size]
    states = states[:, :2]

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    # fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    #ax = fig.axes
    #data = pd.DataFrame(d)
    with sns.axes_style("white"):
        if heatmap:
            num_ticks = 9
            yticks = np.linspace(0, len(ys) - 1, num_ticks, dtype=int)
            # the content of labels of these yticks
            yticklabels = np.array(["%.1f" % ys[idx] for idx in yticks])
            xticks = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
            # the content of labels of these yticks
            xticklabels = np.array(["%.1f" % xs[idx] for idx in xticks])
            fig = plt.figure(figsize=(13, 8))
            data_heatmap = data_heatmap.T
            JG2 = (sns.heatmap(data=data_heatmap, vmin=0, yticklabels=yticklabels, xticklabels=xticklabels))
            JG2.set_yticks(yticks)
            JG2.set_xticks(xticks)
            fig.savefig(base_dir + '/heatmap_map' + str(iter) + '.png')
            plt.close(fig)
            del fig

        # sample less states
        if do_density_map:
            statesplot = states[::100]
            JG = (
                sns.jointplot(
                    x=statesplot[:, 0], 
                    y=statesplot[:, 1], 
                    kind="hex", # { “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” } 
                    color="k",
                    gridsize=500,
                    xlim=(min_state, max_state),
                    ylim=(min_y, max_y)
                ).set_axis_labels("x", "y")
            )
            fig = JG.fig
            JG.savefig(base_dir + '/density_map' + str(iter) + '.png')
            #fig.savefig(base_dir + '/density_map' + str(iter) + '.png')
            plt.close(fig)
            states = sampled_states[(iter) * num_trains * batch_size: (iter + 1) * num_trains * batch_size]
            states = states[:, :2]
            del JG
            JG = (
                sns.jointplot(
                    x=statesplot[:, 0],
                    y=statesplot[:, 1],
                    kind="hex",
                    color="k",
                    gridsize=500,
                    xlim=(min_state, max_state),
                    ylim=(min_y, max_y)
                ).set_axis_labels("x", "y")
            )
            fig = JG.fig
            JG.savefig(base_dir + '/density_map_current_iter_' + str(iter) + '.png')
            plt.close(fig)

            fig = JG.fig


    # fig.legend()
    # if args.show:
    #     plt.show()
    iter += delta_iter
