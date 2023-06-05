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
import glob

from multiprocessing import Pool

# import from repo
from utils.env_utils import env_producer

## path
seed = 0
save = True
sampled = True
base_dir = './data/cliff_mono/sac_/rsx6'

path = os.path.join(base_dir, '*')
#path = os.path.join(base_dir, '*', '*') # Grid search
paths = glob.glob(path)

# parameters
## print corners of the heatma
print_corners = False

## DEBUG
Debug = True
alt = '' # '_str'
particle = False

# Define which iterations to compute the heatmap of #####
itr_list = False

if not itr_list:
	start_itr = 0
	end_itr = 249
	gap = 1

	# start_itr = 30
	# end_itr = 245
	# gap = 5

	end_itr = end_itr + gap
	itrs = np.arange(start_itr, end_itr, gap)

if itr_list: 
	itrs = np.array([
		1
	])
	#itrs = np.array([320])

# if Debug and len(itrs) != 1:
# 	raise ValueError('DEBUG mode cannot generate >1 heatmap')

def plot_heatmaps(base_dir):
	## if == 0 fixed on first std max
	## if < 0 free scale
	vmax_std = -1

	variant = json.load(open(os.path.join(base_dir, 'variant.json')))
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
	env_args['dim'] = (12,) # (variant['dim'],) # 25
	expl_env = env_producer(domain, seed, **env_args)
	eval_env = env_producer(domain, seed * 10 + 1, **env_args)
	obs_dim = expl_env.observation_space.low.size
	action_dim = expl_env.action_space.low.size

	# Get producer function for policy and value functions
	alg = variant['alg']
	trainer = build_variant(variant, return_replay_buffer=False, return_collectors=False)['trainer']

	histog_arr = np.zeros(shape=[len(itrs), 50, 40], dtype=np.int32)

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

	opt_qf = np.load('./data/cliff_mono/opt_qf.npy')
	opt_qf = opt_qf.T
	policy_star = np.load('./data/cliff_mono/opt_ps_det.npy')

	# loop over iterations
	for itr_i, itr in enumerate(itrs):
		# restore snapshot
		experiment = base_dir + '/itr_' + str(itr) + '.zip_pkl'
		exp = load_gzip_pickle(experiment)
		trainer.restore_from_snapshot(exp['trainer'])

		# heatmaps
		data_heatmap = np.zeros((len(xs), len(ya)))
		mean_heatmap = np.zeros((len(xs), len(ya)))
		upbd_heatmap = np.zeros((len(xs), len(ya)))
		#diff_heatmap = np.zeros((len(xs), len(ya)))
		if particle: 
			range_heatmap = np.zeros((len(xs), len(ya)))
		opt_policy_ce = np.zeros(len(xs)) # opt policy current q estimate
		tar_policy = np.zeros(len(xs))
		policy = np.zeros(len(xs))
		opt_policy_ub = np.zeros(len(xs))
		max_std = np.zeros(len(xs))

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

					# ob_t = np.array(ob)
					# # action = np.array(action)
					# ob_t = torch_ify(ob_t)
					# ac_t = torch_ify(ac)
					# if trainer.share_layers:
					# 	raise ValueError('Not implemented')
					# 	qs_t = trainer.q_target(ob_t, ac_t)
					# 	std_t = np_ify(qs_t[:, 1].unsqueeze(-1))
					# else:
					# 	try:
					# 		titleJG21 = 'prv_std - std'
					# 		std_t = np_ify(trainer.prv_std(ob_t, ac_t))
					# 	except:
					# 		titleJG21 = 'target_std - std'
					# 		std_t = np_ify(trainer.std_target(ob_t, ac_t))
				
				if print_corners:
					if (o == 0 and a == -1) or (o == 0 and a == 1) or (o == 25 and a == -1) or (o == 25 and a == 1):
						print('s: %d, a: %d' % (o, a))
						print('mean: %f, std:%f' % (np_ify(mean)[0], np_ify(std)[0]))
				data_heatmap[i, j] = std
				mean_heatmap[i, j] = mean
				upbd_heatmap[i, j] = upper_bound
				#diff_heatmap[i, j] = std_t - std
				if alg in ['p-oac']:
					range_heatmap[i, j] = maxdmin/std # if ~2 split	p, if >2 distributed
			# optimal policy on this state
			opt_policy_ce[i] = np.argmax(mean_heatmap[i])
			opt_policy_ub[i] = np.argmax(upbd_heatmap[i])
			max_std[i] = np.argmax(data_heatmap[i])
			# max_diff?
			ac, *_ = trainer.target_policy(torch_ify(np.reshape(xs[i], (1,1))), deterministic=True)
			tar_policy[i] =  np_ify(ac[0,0])
			if Debug:
				debug0 = 0
			ac, *_ = trainer.policy.forward(torch_ify(np.reshape(xs[i], (1,1))), deterministic=True)
			policy[i] = np_ify(ac[0,0])
		
		# # DEBUG without having to recreate heatmap
		#import pickle
		#pickle.dump(data_heatmap, open(base_dir + '/heatmap.pkl', "wb"))
		#data_heatmap = load_gzip_pickle(base_dir + '/heatmap.pkl')

		# sampled states
		#histog = np.histogram2d(a3d[itr,:,0],a3d[itr,:,1], bins=[50,40], range=[[0, 25], [-1, 1]])[0]
		if sampled:
			histog = np.histogram2d(a3d[itr,:,0],a3d[itr,:,1], bins=[50,40], range=[[-1, 1], [-1, 1]])[0]
			if not particle:
				histog_arr[itr_i] = histog
				cum_histo = np.sum(histog_arr, axis=0)
			histog = histog.T
			if not particle:
				cum_histo = cum_histo.T
			#histog_ori = histog
			histog = np.log2(histog, out=np.zeros_like(histog), where=(histog!=0))
			cum_histo = cum_histo.astype('float64')
			cum_histo = np.log2(cum_histo, out=np.zeros_like(cum_histo), where=(cum_histo!=0))

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
		upbd_heatmap = upbd_heatmap.T
		#diff_heatmap = diff_heatmap.T

		if particle:
			range_heatmap = range_heatmap.T

		fig, ax = plt.subplots(3,2,figsize=(12, 9))

		fig.patch.set_facecolor('#E0E0E0')

		# xtl = 10
		xtl = 4
		# nxt = int(len(xs)/xtl)+1
		if vmax_std == 0:
			vmax_std = np.max(data_heatmap)
		if vmax_std >= 0:
			JG10 = sns.heatmap(data=data_heatmap, xticklabels = xtl, yticklabels = 8, vmin=0, vmax=vmax_std, cmap="jet_r", ax=ax[1,0])
		else:
			JG10 = sns.heatmap(data=data_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,0])
		JG00 = sns.heatmap(data=mean_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[0,0])
		JG20 = sns.heatmap(data=upbd_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[2,0])
		#JG21 = sns.heatmap(data=diff_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[2,1])

		if sampled:
			if particle:
				JG11 = sns.heatmap(data=range_heatmap, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,1])
			else:
				JG11 = sns.heatmap(data=cum_histo, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,1])
				# JG11 = sns.heatmap(data=histog_ori, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[1,1])

		fig.suptitle(base_dir + 'itr' + str(itr))

		JG10.set_title('Std')
		JG10.set_xticklabels(np.arange(0,ori_max_state+1))
		#JG10.set_xticklabels(np.round(np.linspace(0,ori_max_state,nxt)))
		JG10.set_yticklabels(np.round(np.linspace(-10,10,6))/10)

		JG00.set_title('Mean')
		JG00.set_xticklabels(np.arange(0,ori_max_state+1))
		#JG00.set_xticklabels(np.round(np.linspace(0,ori_max_state,nxt)))
		JG00.set_yticklabels(np.round(np.linspace(-10,10,6))/10)

		JG20.set_title('Upper Bound')
		JG20.set_xticklabels(np.arange(0,ori_max_state+1))
		#JG20.set_xticklabels(np.round(np.linspace(0,ori_max_state,nxt)))
		JG20.set_yticklabels(np.round(np.linspace(-10,10,6))/10)

		# JG21.set_title(titleJG21)
		# JG21.set_xticklabels(np.round(np.linspace(0,ori_max_state,nxt)))
		# JG21.set_yticklabels(np.round(np.linspace(-10,10,6))/10)

		if sampled: 
			JG01 = sns.heatmap(data=histog, xticklabels = xtl, yticklabels = 8, cmap="jet_r", ax=ax[0,1])
			JG01.set_title('sampled [log_2]')
			JG01.set_xticklabels(np.arange(0,ori_max_state+1))
			JG01.set_yticklabels(np.round(np.linspace(-10,7.5,5))/10)

			if particle:
				JG11.set_title('(min - max)/std')
				JG11.set_xticklabels(np.arange(0,26)) # wrong
				JG11.set_yticklabels(np.round(np.linspace(-10,10,6))/10)
			else:
				JG11.set_title('Cum sampled [log_2] WARNING: of plotted heatmaps')
				#JG11.set_title('sampled')
				JG11.set_xticklabels(np.arange(0,ori_max_state+1))
				JG11.set_yticklabels(np.round(np.linspace(-10,7.5,5))/10)

		JG00b = sns.lineplot(x=np.linspace(0,len(xs),len(xs)),y=opt_policy_ce, color='white', ax=ax[0,0])
		JG00c = sns.lineplot(x=np.linspace(0,len(xs),len(xs)),y=((tar_policy+1)*(len(ya)-1)/2), color='black', ax=ax[0,0])

		JG10b = sns.lineplot(x=np.linspace(0,len(xs),len(xs)),y=max_std, color='white', ax=ax[1,0])

		JG20b = sns.lineplot(x=np.linspace(0,len(xs),len(xs)),y=opt_policy_ub, color='white', ax=ax[2,0])
		JG20c = sns.lineplot(x=np.linspace(0,len(xs),len(xs)),y=((policy+1)*(len(ya)-1)/2), color='black', ax=ax[2,0])
		
		JG21b = sns.lineplot(x=np.linspace(0,61,61),y=policy_star*(opt_qf.shape[0]-1), color='white', ax=ax[2,1]) 

		max_state = 12
		max_action, min_action = +1, -1
		xtl2 = int((opt_qf.shape[1] - 1) / max_state - 0)
		d_y = 0.5
		ytl2 = int((opt_qf.shape[0] - 1) / ((max_action - min_action) / d_y))
		JG21 = sns.heatmap(opt_qf, xticklabels = xtl2, yticklabels = ytl2, cmap="jet_r", ax=ax[2,1])
		JG21.set_xticklabels(np.arange(0,max_state+1))
		JG21.set_yticklabels(np.round(np.linspace(-10,10,int((max_action - min_action)/d_y+1)))/10)

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
		plt.close()

if __name__ == '__main__':
	# plot_heatmaps(paths[0])
	# print(paths)
	
	if not Debug: 
		with Pool(len(paths)) as p:
			p.map(plot_heatmaps, paths)
	else:
		print(paths[0])
		plot_heatmaps(paths[0])





















