import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys

from os.path import dirname as up
repo_dir = up(up(os.path.realpath(__file__)))
sys.path.append(repo_dir)

# parameters 
show = False
save = not show
separate = True

save_inside = separate
max_rows = 3000
suffix = ''

small_scale = False
grads = False
if small_scale:
	suffix = suffix + '-scale'
if grads:
	suffix = suffix + '-grads'

# code
def running_mean(x, N):
	divider = np.convolve(np.ones_like(x), np.ones((N,)), mode='same')
	return np.convolve(x, np.ones((N,)), mode='same') / divider
env_to_bounds = {
	'point/hard': (-6000, -1000),
	'point/maze': (-10000, -5000),
	'point/maze_easy': (-8000, -4000),
}

#dir = '../data/data_remote/'
dir = os.path.join(repo_dir, 'data')
#envs = ['riverswim'] #,,'cartpole', 'mountain', 'riverswim'

envs = [
	#'point/server'
	#'point/final/MS_gswac_tun2_ub'
	#'point/final/M_oac_SR_tuning'
	#'point/final2/MS_OAC_tun2'
	#'point/final2'
	#'point/final/MS_gswac_tun2_mu'
	#'point/final/DL_gswac_tun2_mu'
	#'point/final3'
	#'point/final3/Medium'
	#'point/final3/Maze_simple'
	#'point/final3/Double_L'
	#'point/para/terminal'
	#'point/final3/Maze_simple_SR'
	#'point/final3'
	#'point/final3/Double_L_SR'
	#'humanoid/mean_update_/gs-oac_'
	#'point/double_I/terminal/mean_update_/gs-oac_'
	#'point/double_I/terminal/sac_'
	'point/maze_simple/terminal/mean_update_/gs-oac_'
]

settings = [
	'MS5_GSWACmu'

	#'DI4_GSWACmu'

	# 'hWAC_qty-0.6',
	# 'hWAC_qty-1',
	# 'hWAC_qty-1.5'
	#'oac_',
	#'sac_end'
	# 'OAC',
	# 'GSWACubu1000',
	# 'GSWACubu',
	# #'GSWACmu',
	# 'OAC_SR_400',
	# 'SAC_SR_400',
	
	# 'M3_SAC',
	# 'M3_OAC',
	# 'M3_GSWAC',
	#'GSWACmu',
	# 'DL3_SAC_SR',
	# 'DL3_OAC_SR',
	# 'DL3_GSWAC_SR',
	#'DL3_gswac_mu_q1w0.6'
	#'DL3_GSWACubu1000_SR'
	
	#'MS3_gswac_mu_q0.6w0.6'
	# 'DL3_OAC'
	# 'DL3_SAC'
	# 'MS3_OAC'
	# 'MS3_SAC'

	# 'gs-oac_/PARA_GSWACubu',
	# 'gs-oac_/PARA_GSWACubu_SR',
	# 'mean_update_/gs-oac_/PARA_GSWACmu',
	# 'mean_update_/gs-oac_/PARA_GSWACmu_SR',
]

colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']

if grads:
	fields = [
		'exploration/Average Returns', 
		'remote_evaluation/Average Returns',
		'trainer/QF mean', 
		#'trainer/QF std',
		'trainer/Critic Target Action Grad PTH',
		#'trainer/Policy Loss',
		#'trainer/Policy Grad',
		'trainer/Target Policy Grad',
		'trainer/Critic Target Action Grad',
		#'trainer/Q FSTD Predictions Mean', 
		#'remote_evaluation/Num Paths',
		#'trainer/' + 'Q Loss'
		'trainer/weights norm',
		'trainer/weights bias',
	]  # 'trainer/QF std 2']
else:
	fields = [
		'exploration/Average Returns', 
		'remote_evaluation/Average Returns',
		'trainer/QF mean', 
		'trainer/QF std',
		'trainer/Critic Target Action Grad PTH',
		#'trainer/Policy Loss',
		#'trainer/Policy Grad',
		#'trainer/Target Policy Grad',
		#'trainer/Critic Target Action Grad',
		'trainer/Q FSTD Predictions Mean', 
		'exploration/Num Paths',
		'remote_evaluation/Num Paths',
		#'trainer/Alpha'
		#'trainer/' + 'Q Loss'
		#'trainer/weights norm',
		#'trainer/weights bias'
	]  # 'trainer/QF std 2']

#
# 'exploration/Returns Max', 'remote_evaluation/Returns Max',
# 'trainer/Policy mu Mean', 'trainer/Policy log std Mean']
field_to_label = {
	'remote_evaluation/Average Returns': 'offline return',
	'exploration/Average Returns': 'online return',
	'trainer/QF mean': 'Mean Q',
	'trainer/QF std': 'Std Q',
	'trainer/QF std 2': 'Std Q 2',
	'trainer/QF Unordered': 'Q Unordered samples',
	'trainer/QF target Undordered': 'Q Target Unordered samples',
	'exploration/Returns Max': 'online Max Return',
	'remote_evaluation/Returns Max': 'offline Max Return',
	'trainer/Policy mu Mean': 'policy mu',
	'trainer/Policy log std Mean': 'policy std',
	'trainer/Policy Loss': 'policy loss',
	'trainer/Target Policy Loss': 'target policy loss',
	'trainer/' + 'QF' + str(7) + ' Loss': 'upper bound loss',
	'trainer/' + 'QF1 Loss': 'upper bound loss',
	'remote_evaluation/Num Paths': 'offline num of episodes',
	'trainer/Q Loss': 'Critic Loss',
	'trainer/Q FSTD Predictions Mean': 'Std of fake sample', 
	'trainer/Q FSTD Target Mean': 'Std target of fake samples',
	'trainer/Policy Grad': 'Policy Gradient',
	'trainer/Target Policy Grad': 'Target Policy Gradient',
	'trainer/Critic Target Action Grad': 'Critic Target Action Grad',
	'trainer/Critic Target Action Grad PTH': 'Critic Target Action Grad PTH',
	'trainer/weights norm': 'weights norm',
	'trainer/weights bias': 'weights bias',
	'exploration/Num Paths': 'expl num of episodes',
	'trainer/Alpha': 'Alpha'
}
count = 0
plot_count = 0
n_col = 2
subsample = 1
for env in envs:
	fig, ax = plt.subplots(int(np.ceil(len(fields) / n_col)), n_col, figsize=(12, 9.5))
	fig.suptitle(env)
	col = 0
	for f, field in enumerate(fields):
		for s, setting in enumerate(settings):
			#path = os.path.join(dir, env, setting, '*', 'progress.csv')
			path = dir + '/' + env + '/' + setting + '/*/progress.csv'
			paths = glob.glob(path)
			print("Path:", path)
			# print("Paths:", paths)
			min_rows = np.inf

			results = []
			final_results = []
			for j, p in enumerate(paths):
				print("p:", p)
				try:
					data = pd.read_csv(p, usecols=[field], header=0)

					#print(data)
				except:
					if field == 'trainer/' + 'QF' + str(7) + ' Loss':
						try:
							data = pd.read_csv(p, usecols=['trainer/' + 'QF1 Loss'])
						except:
							break
					else:
						break
				if len(data) > max_rows:
					data = data[:max_rows]

				try:
					res = np.array(data[field], dtype=np.float64)
				except:
					res = np.array(data[field], dtype=np.float64)
				try:
					res = np.array(data[field], dtype=np.float64)
				except:
					try:
						res = np.array(data['trainer/' + 'QF1 Loss'], dtype=np.float64)
					except:
						print("What")
				if separate:
					if f == 0:
						label = setting  + '-' + str(p)
					else:
						label = None
					mean = running_mean(res, subsample)
					x = list(range(len(mean)))
					ax[col // n_col][col % n_col].plot(x, mean, label=label, color=colors[j])
					#plt.plot(res, label=setting + '-' + str(j), color=colors[count % len(colors)])
					count += 1
				else:
					if len(res) < min_rows:
						min_rows = len(res)
					results.append(res)
			if not separate and len(results) > 0:
				print(len(results))
				if min_rows > max_rows:
					min_rows = max_rows
				for i, _ in enumerate(paths):
					final_results.append(results[i][:min_rows])
				data = np.stack(final_results, axis=0)
				n = data.shape[0]
				#print(data)
				# mean = np.median(data, axis=0)
				mean = np.mean(data, axis=0)
				std = np.std(data, axis=0)
				mean = running_mean(mean, subsample)
				std = running_mean(std, subsample)
				x = list(range(len(mean)))
				#indexes = [i * subsample for i in range(len(mean) // subsample)]
				# mean = mean[indexes]
				# std = std[indexes]
				#x = indexes
				if f == 0:
					label = setting
				else:
					label = None
				ax[col // n_col][col % n_col].plot(x, mean, label=label, color=colors[s])
				if n > 1:
					ax[col // n_col][col % n_col].fill_between(x, mean - 2 * (std / np.sqrt(n)),
										 mean + 2 * (std / np.sqrt(n)),
								 alpha=0.2, color=colors[s])
		ax[col // n_col][col % n_col].set_title(field_to_label[field], fontdict={'fontsize': 7})
		#if field in ['remote_evaluation/Average Returns', 'exploration/Average Returns'] and 'point' in env:
		#	bounds = env_to_bounds[env]
		#	ax[col // n_col][col % n_col].set_ylim(bounds)
		if field in ['trainer/Q Loss'] and 'point' in env:
			ax[col // n_col][col % n_col].set_ylim((0, 20))
		if field in ['trainer/Target Policy Grad',]:
			ax[col // n_col][col % n_col].set_ylim((0, 20))
			if small_scale:
				ax[col // n_col][col % n_col].set_ylim((0, 0.04))
		if field in ['trainer/Critic Target Action Grad']:
			ax[col // n_col][col % n_col].set_ylim((0, 5))
			if small_scale:
				ax[col // n_col][col % n_col].set_ylim((0, 0.04))
		if field in ['trainer/Critic Target Action Grad PTH']:
			ax[col // n_col][col % n_col].set_ylim((0, 2.5))
			if small_scale:	
				ax[col // n_col][col % n_col].set_ylim((0, 0.2))	
		if col // n_col == int(np.ceil(len(fields) / n_col)) - 1:
			ax[col // n_col][col % n_col].set_xlabel('epoch', fontdict={'fontsize': 7})
		ax[col // n_col][col % n_col].set_title(field_to_label[field], fontdict={'fontsize': 7})
		# if col // n_col in [0, 2]:
		#	 ax[col // n_col][col % n_col].set_ylim((-6000, -1000))
		col += 1
		plot_count += 1
	fig.legend(loc='lower center', ncol=max(len(settings)//2, 1))
	#fig.savefig(env + '.png')
	if show:
		plt.show(block=False) # remove block=False if you prefer closing the window manually
	if show:
		input("Press Enter to continue...")
	if save:
		if save_inside: 
			output_name = os.path.join(dir, env, settings[0], settings[0])
		else:
			output_name = os.path.join(dir, env, os.path.basename(os.path.normpath(env)))
		if separate:
			output_name = output_name + '-plot-sep' + suffix + '.png'
		else:
			output_name = output_name + '-plot' + suffix + '.png'
		print("output " + output_name)
		fig.savefig(output_name)

# oac = pd.read_csv('oac/1581679951.0551817/progress.csv')
# w_oac = pd.read_csv('../data/riverswim/oac-w/1581679961.4996805/progress.csv')
# sac = pd.read_csv('../data/riverswim/sac/1581692438.1551993/progress.csv')
#
# oac_res = np.array(oac['remote_evaluation/Average Returns'])
# w_oac_res = np.array(w_oac['remote_evaluation/Average Returns'])
# sac_res = np.array(sac['remote_evaluation/Average Returns'])
# plt.plot(oac_res, label='oac')
# plt.plot(w_oac_res, label='oac-w')
# plt.plot(sac_res, label='sac')

plt.close('all')

