
import os 
import numpy as np
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

alg = 'gs-oac'

hp_names = {
	'deltas': 'delta',
}

hp_paths = {
	'deltas': 'delta', 
}

if alg == 'gs-oac':
	#env = './data/lqg/gs-oac_/lq18qtysB/'
	#env = './data/lqg/gs-oac_/lq18qtys'
	#env = './data/riverswim/25/gs-oac_/rs23t1'
	env = './data/point/final3/RS3delta'
	
	alt = ''

	# main_hp0 = 0.5
	# main_hp1 = 0.5

	settings_h = {
		'deltas': [0.5, 0.55, 0.65, 0.75, 0.85, 0.95],
		#'deltas': [0.75],
		# 'qs': [0],
		# 'ws': [0]
	}	

if alg == 'oac':
	#env = './data/lqg/oac_/lq20oacB'
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
	#hp1 = list(settings_h.keys())[1]

	# scores = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])
	# stds = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])

	scores = np.zeros(shape=[len(settings_h[hp0])])
	stds = np.zeros(shape=[len(settings_h[hp0])])

	delta = 0.001

	for i, hp0v in enumerate(settings_h[hp0]):
		#for j, hp1v in enumerate(settings_h[hp1]):
			
		# if (hp0v == 0 and hp1v != 0) or (hp0v != 0 and hp1v == 0):
		# 	scores[i, j] = scores[0,0]
		# 	continue
		
		setting = f'{hp_paths[hp0]}{hp0v}' 
		path = os.path.join(env, setting, '*')
		all_seeds = glob.glob(path)

		seed_res = np.zeros(shape=[len(all_seeds)], dtype=np.float32)
		for s_idx, seed in enumerate(all_seeds):
			print(f'path: {seed}')
			variant = json.load(open(os.path.join(seed, 'variant.json')))

			states = np.array(pd.read_csv(os.path.join(seed, 'sampled_states.csv'), header=None).iloc[:, 0])
			actions = np.array(pd.read_csv(os.path.join(seed, 'sampled_actions.csv'), header=None).iloc[:, 0])
			tot = len(states)

			a2d = np.stack((states, actions), axis=1, out=None)
			epochs = variant['algorithm_kwargs']['num_epochs'] 
			a3d = np.reshape(a2d, (epochs,int(tot/epochs),2))

			ac_bins = 20
			ob_bins = 20

			histog_arr = np.zeros(shape=[epochs, ob_bins, ac_bins], dtype=np.int32)
			res_arr = np.zeros(shape=[epochs], dtype=np.float32)

			# ASK 
			# should the histogram be normalized based on the amount of bins?

			for itr_i in range(epochs):
				
				if False: # normalized
					a_o_range = 1 - (-1) 
					bin_dim = (ac_bins / a_o_range) * (ob_bins / a_o_range)
					histog = np.histogram2d(a3d[itr_i,:,0],a3d[itr_i,:,1], bins=[20,20], range=[[-1, 1], [-1, 1]], density=True)[0]/bin_dim

				histog = np.histogram2d(a3d[itr_i,:,0],a3d[itr_i,:,1], bins=[ob_bins,ob_bins], range=[[-1, 1], [-1, 1]])[0] 
				histog_arr[itr_i] = use_histo = histog
				# normalize for total amount of samples
				if use_cum_histo:
					use_histo = np.sum(histog_arr, axis=0)/np.sum(histog_arr)

				# compute the amount of bins with more than certain threshold
				# save the number in array
				res_arr[itr_i] = np.sum(np.where(use_histo > delta, 1, 0))/(ac_bins * ob_bins)
			
			seed_res[s_idx] = np.mean(res_arr)
			df = pd.DataFrame([np.mean(res_arr)], columns=['index'])
			df.to_csv(os.path.join(seed, f'index_d{delta}.csv'))
		# scores[i, j] = np.mean(seed_res)
		# stds[i, j] = np.var(seed_res)

		print(f'done: {hp0}={hp0v}')

	print('done')

main_function()















