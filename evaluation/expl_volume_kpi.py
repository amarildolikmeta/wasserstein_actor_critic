
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

LEGEND_FONT_SIZE = 28
AXIS_FONT_SIZE = 40
TICKS_FONT_SIZE = 40
MARKER_SIZE = 10
LINE_WIDTH = 3.0
TITLE_SIZE= 28

colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']

alg = 'oac'

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
	main_hp1 = 0.5

	settings_h = {
		'qs': [0, 0.25, 0.5, 0.75, 1],
		'ws': [0, 0.25, 0.5, 0.75, 1]
		# 'qs': [0.5],
		# 'ws': [0.25, 0.5]
	}

if alg == 'oac':
	#env = './data/lqg/oac_/lq20oacB'
	#env = './data-plot/lq20oac'
	env = './data/riverswim/25/oac_/rs21oac'

	alt = ''

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
	hp1 = list(settings_h.keys())[1]

	scores = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])
	stds = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])
	num_seeds = np.zeros(shape=[len(settings_h[hp0]), len(settings_h[hp1])])

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
				seed_res[s_idx] = pd.read_csv(os.path.join(seed, f'index_d{delta}.csv'))['index'][0]

			print(f'done: {hp0}={hp0v}, {hp1}={hp1v}')
			scores[i, j] = np.mean(seed_res)
			stds[i, j] = np.std(seed_res)
			num_seeds[i, j] = len(all_seeds)
	
	# hp1
	main_hp1_idx = settings_h[hp1].index(main_hp1)
	scores_hp0 = scores[:, main_hp1_idx]
	stds_hp0 = stds[:, main_hp1_idx]

	# hp0
	main_hp0_idx = settings_h[hp0].index(main_hp0)
	scores_hp1 = scores[main_hp0_idx, :]
	stds_hp1 = stds[main_hp0_idx, :]

	fig, ax = plt.subplots(1, 1, figsize=(12, 12))

	# FIRST PLOT
	# ax.set_title(f'Varying {hp_names[hp0]}, {hp_names[hp1]}={main_hp1}')
	plot_list = []
	label_list = []
	for j, hp1v in enumerate(settings_h[hp1]):
		plot_list.append(
			ax.plot(
				settings_h[hp0],
				scores[:, j],
				label=hp1v, 
				linewidth=LINE_WIDTH,
				color=colors[j], 
				marker=markers[j],
				markersize=MARKER_SIZE
			)[0]
		)
		label_list.append(hp1v)
		ax.fill_between(
			settings_h[hp0], 
			scores[:, j] - 2 * (stds[:, j] / np.sqrt(num_seeds[:, j])), 
			scores[:, j] + 2 * (stds[:, j] / np.sqrt(num_seeds[:, j])), 
			alpha=0.2, 
			color=colors[j]
		) 

	# ax.plot(settings_h[hp0], scores_hp0)

	# ax.fill_between(settings_h[hp0], scores_hp0 - 2 * (stds_hp0 / np.sqrt(4)), scores_hp0 + 2 * (stds_hp0 / np.sqrt(4)), alpha=0.2)
	#ax.set_xlabel(f'{hp_names[hp0]}', fontsize=AXIS_FONT_SIZE)
	ax.set_xlabel(r'$\beta_{UB}$', fontsize=AXIS_FONT_SIZE)
	ax.set_ylabel(' ', fontsize=AXIS_FONT_SIZE)
	ax.set_xticks(settings_h[hp0])
	y_ticks = [0.05, 0.1, 0.15, 0.2] # RS
	#y_ticks = [0.3, 0.4, 0.5 0.6] # LQG
	ax.set_yticks(y_ticks)
	x = np.linspace(0, len(scores_hp1), len(scores_hp1))

	ax.xaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	ax.yaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	ax.tick_params(labelsize=TICKS_FONT_SIZE)
	ax.grid(linestyle=":")

	# lgd = fig.legend(loc='lower center', ncol=len(settings_h[hp1]), title=hp_names[hp1], fontsize=LEGEND_FONT_SIZE)
	# plt.setp(lgd.get_title(),fontsize=LEGEND_FONT_SIZE)

	legendFig = plt.figure("Legend plot")
	legendFig.legend(plot_list, label_list, loc='center', ncol=10, title=r'$\delta$')
	output_name_lgd = 'lgd_delta'
	legendFig.savefig(os.path.join(env, output_name_lgd + alt + '.png'))
	legendFig.savefig(os.path.join(env, output_name_lgd + alt + '.pdf'), format='pdf')
	#legendFig.savefig(output_name_lgd + alt + '.pdf', format='pdf')

	fig.savefig(os.path.join(env, output_name + 'fixed_delta' + alt + '.png'),bbox_inches='tight')
	fig.savefig(os.path.join(env, output_name + 'fixed_delta' + alt + '.pdf'),bbox_inches='tight' , format='pdf')
	plt.close('all') 

	plt.clf()
	fig, ax = plt.subplots(1, 1, figsize=(12, 12))

	# second plot
	# ax.set_title(f'Varying {hp_names[hp1]}, {hp_names[hp0]}={main_hp0}')
	plot_list = []
	label_list = []
	for i, hp0v in enumerate(settings_h[hp0]):
		plot_list.append(
			ax.plot(
				settings_h[hp1],
				scores[i, :],
				label=hp0v,
				linewidth=LINE_WIDTH,
				color=colors[i], 
				marker=markers[i],
				markersize=MARKER_SIZE
			)[0]
		)
		label_list.append(hp0v)
		ax.fill_between(
			settings_h[hp1],
			scores[i, :] - 2 * (stds[i, :] / np.sqrt(num_seeds[i, :])),
			scores[i, :] + 2 * (stds[i, :] / np.sqrt(num_seeds[i, :])),
			alpha=0.2,
			color=colors[i]
		)
	
	#ax.set_xlabel(f'{hp_names[hp1]}', fontsize=AXIS_FONT_SIZE)
	ax.set_xlabel(r'$\delta$', fontsize=AXIS_FONT_SIZE)
	ax.set_ylabel(' ', fontsize=AXIS_FONT_SIZE)
	ax.set_xticks(settings_h[hp1])
	y_ticks = [0.05, 0.1, 0.15, 0.2] # RS
	#y_ticks = [0.3, 0.4, 0.5 0.6] # LQG
	ax.set_yticks(y_ticks)
	ax.xaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	ax.yaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	ax.tick_params(labelsize=TICKS_FONT_SIZE)
	ax.grid(linestyle=":")


	# lgd = fig.legend(ncol=len(settings_h[hp0]), title=hp_names[hp0], fontsize=LEGEND_FONT_SIZE)
	# plt.setp(lgd.get_title(),fontsize=LEGEND_FONT_SIZE)

	legendFig = plt.figure("Legend plot")
	legendFig.legend(plot_list, label_list, loc='center', ncol=10, title=r'$\beta_{UB}$')
	output_name_lgd = 'lgd_beta'
	legendFig.savefig(os.path.join(env, output_name_lgd + alt + '.png'))
	legendFig.savefig(os.path.join(env, output_name_lgd + alt + '.pdf'), format='pdf')

	fig.savefig(os.path.join(env, output_name + 'fixed_beta' + alt + '.png'),bbox_inches='tight')
	fig.savefig(os.path.join(env, output_name + 'fixed_beta' + alt + '.pdf'),bbox_inches='tight' , format='pdf')

	plt.close('all') 
	fig, ax = plt.subplots(1, 1, figsize=(12, 12))
	
	## heatmap
	scores = np.flip(scores, axis=0)
	ax = sns.heatmap(
		data=scores,
	 	xticklabels=settings_h[hp1],
		yticklabels=settings_h[hp0][::-1]
	)
	
	#ax.set_title('Varying Both')

	#ax.set_xlabel(f'{hp_names[hp1]}', fontsize=AXIS_FONT_SIZE)
	ax.set_xlabel(r'$\delta$', fontsize=AXIS_FONT_SIZE)
	#ax.set_ylabel(f'{hp_names[hp0]}', fontsize=AXIS_FONT_SIZE)
	ax.set_ylabel(r'$\beta_{UB}$', fontsize=AXIS_FONT_SIZE)
	ax.xaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	ax.yaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
	cbar = ax.collections[0].colorbar
	# here set the labelsize by 20
	cbar.ax.tick_params(labelsize=TICKS_FONT_SIZE)
	ax.tick_params(labelsize=TICKS_FONT_SIZE)
	#tax.grid(linestyle=":")

	fig.savefig(os.path.join(env, output_name + 'heatmap' + alt + '.png'),bbox_inches='tight')
	fig.savefig(os.path.join(env, output_name + 'heatmap' + alt + '.pdf'),bbox_inches='tight', format='pdf')
	

	print('done')

main_function()















