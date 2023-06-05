
import os
from re import L 
import numpy as np
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')
plt.style.use('default')
plt.rc('font', family='serif')

LEGEND_FONT_SIZE = 28
AXIS_FONT_SIZE = 28
TICKS_FONT_SIZE = 26
MARKER_SIZE = 10
LINE_WIDTH = 3.0
TITLE_SIZE= 28

AXIS_FONT_SIZE = 40
TICKS_FONT_SIZE = 40

colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']

alg = 'gs-oac'

hp_names = {
	'qs': 'quantity',
	'ws': 'weight',
	'beta_s': 'beta',
	'delta_s': 'delta'
}

hp_paths = {
	# 'qs': 'q', 
	# 'ws': 'w',
	# 'beta_s': 'beta',
	'deltas': 'delta'
}

if alg == 'gs-oac':
	#env = './data/lqg/gs-oac_/lq18qtysB/'
	#env = './data/lqg/gs-oac_/lq18qtys'
	#env = './data/riverswim/25/gs-oac_/rs23t1'
	env = './data/point/final3/LQG3delta'
	
	alt = '_pap'

	# main_hp0 = 0.25
	# main_hp1 = 0.5

	settings_h = {
		'deltas': [0.5, 0.55, 0.65, 0.75, 0.85, 0.95],
		#'ws': [0, 0.25, 0.5, 0.75, 1]
		# 'qs': [0.5],
		# 'ws': [0.25, 0.5]
	}

if alg == 'oac':
	env = './data/lqg/oac_/lq20oacB'
	env = './data/lqg/oac_/lq20oac'

	alt = '_altfin'

	main_hp0 = 4
	main_hp1 = 20

	settings_h = {
		'beta_s': [1, 2, 3, 4, 5, 6, 7],
		'delta_s': [5, 10, 20, 30, 50, 70]
	}	

sampled = True
use_cum_histo = True

output_name = 'index'

def main_function():

	n_col = 3

	hp0 = list(settings_h.keys())[0]
	#hp1 = list(settings_h.keys())[1]

	# scores = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])
	# stds = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])
	# num_seeds = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])

	scores = np.zeros(shape=[len(settings_h[hp0])])
	stds = np.zeros(shape=[len(settings_h[hp0])])
	num_seeds = np.zeros(shape=[len(settings_h[hp0])])

	delta = 0.001

	for i, hp0v in enumerate(settings_h[hp0]):
		# hp0v_store = hp0v
		# for j, hp1v in enumerate(settings_h[hp1]):
		# hp0v = hp0v_store
		# if hp0v == 0:
		# 	hp1v = 0
		# elif hp1v == 0:
		# 	hp0v_store = hp0v
		# 	hp0v = 0

		# if (hp0v == 0 and hp1v != 0) or (hp0v != 0 and hp1v == 0):
		# 	scores[i, j] = scores[0,0]
		# 	continue
		
		setting = f'{hp_paths[hp0]}{hp0v}' 
		path = os.path.join(env, setting, '*')
		all_seeds = glob.glob(path)

		seed_res = np.zeros(shape=[len(all_seeds)], dtype=np.float32)
		for s_idx, seed in enumerate(all_seeds):
			seed_res[s_idx] = pd.read_csv(os.path.join(seed, f'index_d{delta}.csv'))['index'][0]

		print(f'done: {hp0}={hp0v}')
		scores[i] = np.mean(seed_res)
		stds[i] = np.std(seed_res)
		num_seeds[i] = len(all_seeds)
	
	# hp1
	#main_hp1_idx = settings_h[hp1].index(main_hp1)
	scores_hp0 = scores
	stds_hp0 = stds

	# hp0
	# main_hp0_idx = settings_h[hp0].index(main_hp0)
	# scores_hp1 = scores[main_hp0_idx, :]
	# stds_hp1 = stds[main_hp0_idx, :]

	fig, ax = plt.subplots(1, 1, figsize=(12, 12))
	ax.plot(
		settings_h[hp0],
		scores[:],
		linewidth=LINE_WIDTH,
		color=colors[0], 
		marker=markers[0],
		markersize=MARKER_SIZE
	)[0]
	ax.fill_between(
		settings_h[hp0], 
		scores[:] - 2 * (stds[:] / np.sqrt(num_seeds[:])), 
		scores[:] + 2 * (stds[:] / np.sqrt(num_seeds[:])), 
		alpha=0.2, 
		color=colors[0]
	)
	ax.set_xlabel(r'$\delta$', fontsize=AXIS_FONT_SIZE)
	ax.set_ylabel('Coverage', fontsize=AXIS_FONT_SIZE)
	#x_labels = ['0.5', '0.55', '0.65', '0.75', '0.85', '0.95']
	x_labels = [0.5, 0.6, 0.7, 0.8, 0.9]
	ax.set_xticks(x_labels)
	#ax.set_xticks(settings_h[hp0])
	#ax.set_xticklabels(x_labels)
	#plt.xticks(rotation=45)
	#y_labels = [0.1, 0.2, 0.3] # RS
	y_labels = [0.5, 0.6, 0.7] # LQG
	ax.set_yticks(y_labels)
	ax.xaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	ax.yaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	ax.tick_params(labelsize=TICKS_FONT_SIZE)
	ax.grid(linestyle=":")

	fig.savefig(os.path.join(env, output_name + 'delta' + '.png'), bbox_inches='tight')
	fig.savefig(os.path.join(env, output_name + 'delta' + '.pdf'), format='pdf', bbox_inches='tight')
	plt.close('all') 

	plt.clf()
	print('done')

main_function()















