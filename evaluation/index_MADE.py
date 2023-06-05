
import os 
import numpy as np
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

alg = 'gs-oac'

hp_names = {
	'qs': 'quantity',
	'ws': 'weight',
	'beta_s': 'beta',
	'delta_s': 'delta'
}

hp_paths = {
	'qs': 'q', 
	'ws': 'w',
	'beta_s': 'beta',
	'delta_s': 'delta'
}

if alg == 'gs-oac':
	#env = './data/lqg/gs-oac_/lq18qtysB/'
	#env = './data/lqg/gs-oac_/lq18qtys'
	#env = './data/riverswim/25/gs-oac_/rs23t1'
	env = './data/point/final3/RS3'
	
	alt = ''

	# main_q = 0.75
	# main_w = 0.5
	main_hp0 = 0.25
	main_hp1 = 0.25

	settings_h = {
		'qs': [0, 0.25, 0.5, 0.75, 1],
		'ws': [0, 0.25, 0.5, 0.75, 1]
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

def main_function():

	n_col = 3
	fig, ax = plt.subplots(1, n_col, figsize=(15, 4))

	hp0 = list(settings_h.keys())[0]
	hp1 = list(settings_h.keys())[1]

	scores = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])
	stds = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])

	delta = 0.001

	for i, hp0v in enumerate(settings_h[hp0]):
		hp0v_store = hp0v
		for j, hp1v in enumerate(settings_h[hp1]):
			hp0v = hp0v_store
			if hp0v == 0:
				hp1v = 0
			elif hp1v == 0:
				hp0v_store = hp0v
				hp0v = 0

			if (hp0v == 0 and hp1v != 0) or (hp0v != 0 and hp1v == 0):
				scores[i, j] = scores[0,0]
				continue
			
			setting = f'{hp_paths[hp0]}{hp0v}_{hp_paths[hp1]}{hp1v}' 
			path = os.path.join(env, setting, '*')
			all_seeds = glob.glob(path)

			seed_res = np.zeros(shape=[len(all_seeds)], dtype=np.float32)
			for s_idx, seed in enumerate(all_seeds):
				
				debug = 0

				progresscsv = os.path.join(seed, 'progress.csv')
				data = pd.read_csv(progresscsv, usecols=['scores/MADE'], header=0)
				res_arr = np.array(data['scores/MADE'], dtype=np.float64)
				
				seed_res[s_idx] = np.mean(res_arr)

			print(f'done: {hp0}={hp0v}, {hp1}={hp1v}')
			scores[i, j] = np.mean(seed_res)
			stds[i, j] = np.var(seed_res)
	
	# hp1
	main_hp1_idx = settings_h[hp1].index(main_hp1)
	scores_hp0 = scores[:, main_hp1_idx]
	stds_hp0 = stds[:, main_hp1_idx]

	# hp0
	main_hp0_idx = settings_h[hp0].index(main_hp0)
	scores_hp1 = scores[main_hp0_idx, :]
	stds_hp1 = stds[main_hp0_idx, :]
	
	scores = np.flip(scores, axis=0)

	# printx
	ax[0].set_title(f'Varying {hp_names[hp0]}, {hp_names[hp1]}={main_hp1}')
	ax[0].errorbar(settings_h[hp0], scores_hp0, yerr=stds_hp0, fmt='-o', ecolor='lightgray')
	#ax[0].errorbar(settings_h[hp0], scores_hp0, fmt='-o', ecolor='lightgray')
	ax[0].set_xlabel(f'{hp_names[hp0]}')	

	ax[0].set_xticks(settings_h[hp0])
	x = np.linspace(0, len(scores_hp1), len(scores_hp1))

	ax[1].set_title(f'Varying {hp_names[hp1]}, {hp_names[hp0]}={main_hp0}')
	ax[1].errorbar(settings_h[hp1], scores_hp1, yerr=stds_hp1, fmt='-o', ecolor='lightgray')
	#ax[1].errorbar(settings_h[hp1], scores_hp1, fmt='-o', ecolor='lightgray')
	ax[1].set_xlabel(f'{hp_names[hp1]}')
	ax[1].set_xticks(settings_h[hp1])
	
	## heatmap
	ax2 = sns.heatmap(data=scores, xticklabels=settings_h[hp1], yticklabels=settings_h[hp0])
	
	ax2.set_title('Varying Both')

	ax2.set_xlabel(f'{hp_names[hp1]}')
	ax2.set_ylabel(f'{hp_names[hp0]}')
	name = 'MADE'
	name = name + alt
	
	name = name + '.png'
	fig.savefig(os.path.join(env, name))
	plt.close('all') 

	print('done')

main_function()















