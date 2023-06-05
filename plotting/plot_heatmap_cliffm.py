# import libraries
import os
import json
import numpy as np
import pandas as pd
from utils.core import np_ify, torch_ify
from utils.pythonplusplus import load_gzip_pickle
from utils.variant_util import build_variant
import seaborn as sns
import matplotlib.pyplot as plt

# import from repo
from utils.env_utils import env_producer

## path
seed = 0
save = False
sampled = True
base_dir = './data/cliff_mono/g-oac_/c004.r-.1+0/s447725'
#base_dir = './data/cliff_mono/g-oac_/c003.2l/s1'
#base_dir = './data/debug/cliff_mono/g-oac_/1638276283.11392'
#base_dir = './data/debug/cliff_mono/g-oac_/02'
#base_dir = './data/cliff_mono/g-oac_/c001.4/00'

variant = json.load(open(os.path.join(base_dir, 'variant.json')))

# parameters
## if == 0 fixed on first std max
## if < 0 free scale
vmax_std = -1
## print corners of the heatmap
print_corners = False

## DEBUG
Debug = False
alt = '' # '_str'
guassian = True
particle = False

# Define which iterations to compute the heatmap of #####
itr_list = False

if not itr_list and not Debug:
	start_itr = 10
	end_itr = 20
	gap = 1
	end_itr = end_itr + gap
	itrs = np.arange(start_itr, end_itr, gap)

if itr_list: 
	itrs = np.array([
		-1
	])
	#itrs = np.array([320])

if Debug and len(itrs) != 1:
	raise ValueError('DEBUG mode cannot generate >1 heatmap')

# states and actions arrays
if sampled:
	states = np.array(pd.read_csv(os.path.join(base_dir, 'sampled_states.csv'), header=None).iloc[:, 0])
	actions = np.array(pd.read_csv(os.path.join(base_dir, 'sampled_actions.csv'), header=None).iloc[:, 0])
	tot = len(states)

	## 2d array
	a2d = np.stack((states, actions), axis=1, out=None)

	## add epochs dimension 3d
	epochs = variant['algorithm_kwargs']['num_epochs'] 
	a3d = np.reshape(a2d, (epochs,int(tot/epochs),2))

# env
domain = variant['domain'] # riverswim
env_args = {}
env_args['dim'] = (variant['dim'],) # 25
expl_env = env_producer(domain, seed, **env_args)
eval_env = env_producer(domain, seed * 10 + 1, **env_args)
obs_dim = expl_env.observation_space.low.size
action_dim = expl_env.action_space.low.size

# Get producer function for policy and value functions
alg = variant['alg']
trainer = build_variant(variant, return_replay_buffer=False, return_collectors=False)['trainer']

# heatmap bounds and steps
#min_state, max_state = eval_env.observation_space.low[0], eval_env.observation_space.high[0] # 0,25
min_state, max_state = -1, 1
min_action, max_action = eval_env.action_space.low[0], eval_env.action_space.high[0]

#delta_state = 0.5
xtl = 4
ori_max_state = 12
delta_state = (max_state - min_state) / ori_max_state / xtl
#delta_state = 2/12/4
delta_action = 0.05
xs = np.array([min_state + i * delta_state for i in range(int((max_state - min_state) / delta_state + 1))])
ya = np.array([min_action + i * delta_action for i in range(int((max_action - min_action) / delta_action + 1))])


histog_arr = np.zeros(shape=[len(itrs), 50, 40], dtype=np.int32)
# loop over iterations
for itr_i, itr in enumerate(itrs):
	# restore snapshot
	experiment = base_dir + '/itr_' + str(itr) + '.zip_pkl'
	exp = load_gzip_pickle(experiment)
	trainer.restore_from_snapshot(exp['trainer'])

	# heatmaps
	data_heatmap = np.zeros((len(xs), len(ya)))
	mean_heatmap = np.zeros((len(xs), len(ya)))
	if particle: 
		range_heatmap = np.zeros((len(xs), len(ya)))

	for i in range(len(xs)): # states
		for j in range(len(ya)): # actions
			o = np.array([xs[i]])
			ob = np.array(o).reshape((1, 1))
			a = np.array([ya[j]])
			ac = np.array(a).reshape((1, a.shape[-1]))

			if alg in ['p-oac']: # compute variance from particles
				qs, _ = trainer.predict(ob, ac, all_particles=True)
				qs = np_ify(qs)
				qs = qs.reshape((1, -1))
				qs = np.sort(qs, axis=1)
				std = np.std(qs.flatten())
				maxdmin = np.max(qs.flatten()) - np.min(qs.flatten())
				mean = np.mean(qs.flatten())
			elif alg in ['g-oac']:
				qs, upper_bound = trainer.predict(ob, ac, std=True)
				std = np_ify(qs[1])[0]
				mean = qs[0]

			if print_corners:
				if (o == 0 and a == -1) or (o == 0 and a == 1) or (o == 25 and a == -1) or (o == 25 and a == 1):
					print('s: %d, a: %d' % (o, a))
					print('mean: %f, std:%f' % (np_ify(mean)[0], np_ify(std)[0]))
			data_heatmap[i, j] = std
			mean_heatmap[i, j] = mean
			if alg in ['p-oac']:
				range_heatmap[i, j] = maxdmin/std # if ~2 split	p, if >2 distributed	  	
	

	# # DEBUG without having to recreate heatmap
	#import pickle
	#pickle.dump(data_heatmap, open(base_dir + '/heatmap.pkl', "wb"))
	#data_heatmap = load_gzip_pickle(base_dir + '/heatmap.pkl')

	# visted states
	#histog = np.histogram2d(a3d[itr,:,0],a3d[itr,:,1], bins=[50,40], range=[[0, 25], [-1, 1]])[0]
	if sampled:
		histog = np.histogram2d(a3d[itr,:,0],a3d[itr,:,1], bins=[50,40], range=[[-1, 1], [-1, 1]])[0]
		if not particle:
			histog_arr[itr_i] = histog
			cum_histo = np.sum(histog_arr, axis=0)
		histog = histog.T
		if not particle:
			cum_histo = cum_histo.T
	# plot
	num_ticks = 9
	yticks = np.linspace(0, len(ya) - 1, len(ya), dtype=int)
	# the content of labels of these yticks
	yticklabels = np.array(["%.1f" % ya[idx] for idx in yticks])
	xticks = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
	# the content of labels of these yticks
	xticklabels = np.array(["%.1f" % xs[idx] for idx in xticks])
	#fig = plt.figure(figsize=(13, 8))
	data_heatmap = data_heatmap.T
	mean_heatmap = mean_heatmap.T
	if particle:
		range_heatmap = range_heatmap.T

	fig, ax = plt.subplots(2,2,figsize=(20, 8))

	# ## differnt color depeding on range
	# plt.axes()
	# max_hm = np.max(data_heatmap)
	# if (max_hm < 2.5):
	# 	JG2 = sns.heatmap(data=data_heatmap, xticklabels = 2, yticklabels = 4, vmin=0, vmax=2.5)
	# else:
	# 	JG2 = sns.heatmap(data=data_heatmap, xticklabels = 2, yticklabels = 4, vmin=0, vmax=8, cmap="YlGnBu_r")

	if vmax_std == 0:
		vmax_std = np.max(data_heatmap)
	if vmax_std >= 0:
		JG2 = sns.heatmap(data=data_heatmap, xticklabels = xtl, yticklabels = 8, vmin=0, vmax=vmax_std, cmap="jet_r", ax=ax[1,0])
	else:
		JG2 = sns.heatmap(data=data_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,0])
	JG3 = sns.heatmap(data=mean_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[0,0])
	
	if sampled:
		if particle:
			JG5 = sns.heatmap(data=range_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,1])
		else:
			JG5 = sns.heatmap(data=cum_histo, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,1])

	fig.suptitle(base_dir + '--' + str(itr))
	
	JG2.set_title('Std')
	JG2.set_xticklabels(np.arange(0,ori_max_state+1))
	JG2.set_yticklabels(np.round(np.linspace(-10,10,6))/10)
	
	JG3.set_title('Mean')
	JG3.set_xticklabels(np.arange(0,ori_max_state+1))
	JG3.set_yticklabels(np.round(np.linspace(-10,10,6))/10)
	
	if sampled: 
		JG4 = sns.heatmap(data=histog, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[0,1])
		JG4.set_title('Visted')
		JG4.set_xticklabels(np.arange(0,13))
		JG4.set_yticklabels(np.round(np.linspace(-10,7.5,5))/10)

		if particle:
			JG5.set_title('(min - max)/std')
			JG5.set_xticklabels(np.arange(0,26))
			JG5.set_yticklabels(np.round(np.linspace(-10,10,6))/10)
		else:
			JG5.set_title('Cum Visted WARNING: of plotted heatmaps')
			JG5.set_xticklabels(np.arange(0,13))
			JG5.set_yticklabels(np.round(np.linspace(-10,7.5,5))/10)

	folder = 'heatmaps' + alt
	if Debug:
		folder = '0debug'

	folder = os.path.join(base_dir, folder)
	if not os.path.exists(folder):
		os.makedirs(folder) 
	output_name = os.path.join(folder, 'hm_' + str(itr) + '.png')
	print(output_name)
	if save:
		fig.savefig(output_name)
	else:
		plt.show()
	plt.close(fig)


















