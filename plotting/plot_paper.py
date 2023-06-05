import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import json
from collections import OrderedDict

from os.path import dirname as up
repo_dir = up(up(os.path.realpath(__file__)))
sys.path.append(repo_dir)

plt.close('all')
plt.style.use('default')
plt.rc('font', family='serif')
#plt.rc('text', usetex=True)

LEGEND_FONT_SIZE = 28
AXIS_FONT_SIZE = 28
TICKS_FONT_SIZE = 26
MARKER_SIZE = 10
LINE_WIDTH = 3.0
TITLE_SIZE= 28

AXIS_FONT_SIZE = 40
TICKS_FONT_SIZE = 40
TITLE_SIZE= 45

linestyles = OrderedDict([('solid', (0, ())),])

# parameters 
show = False
save = not show
save_legend = True
separate = False

top_title = False
xlabel = True
ylabel = False

save_inside = separate
max_rows = 400
suffix = '_ourslegend'

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
dir = os.path.join(repo_dir, 'data-plot')
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
	#'point/final3/Maze_simple'
	#'point/final3/Maze_simple'
	#'Medium'
	#'Medium_SR'
	#'Double_I',
	#'Double_I_SR'
	#'Maze_simple'
	#'Maze_simple_SR'
	#'Double_L'
	'Double_L_SR'
]

settings = [
	# 'SAC',
	# 'OAC',
	# 'GSWACubu',
	#'GSWACmu',
	'SAC_SR_400',
	'OAC_SR_400',
	'GSWACubu1000',
	# 'DL3_SAC_SR',
	# 'DL3_OAC_SR',
	# 'DL3_GSWAC_SR',
	#'DL3_gswac_mu_q1w0.6'
	
	#'MS3_gswac_mu_q0.6w0.6'
	# 'DL3_OAC'
	# 'DL3_SAC'
	# 'MS3_OAC'
	# 'MS3_SAC'
]

colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']


fields = [
	#'exploration/Average Returns', 
	'remote_evaluation/Average Returns',
	# 'trainer/QF mean', 
	# 'trainer/QF std',
	# 'trainer/Critic Target Action Grad PTH',
	# #'trainer/Policy Loss',
	# #'trainer/Policy Grad',
	# #'trainer/Target Policy Grad',
	# #'trainer/Critic Target Action Grad',
	# 'trainer/Q FSTD Predictions Mean', 
	# 'exploration/Num Paths',
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
	'remote_evaluation/Average Returns': 'Average return',
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
	'remote_evaluation/Num Paths': 'Num of episodes',
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

diff_to_name = {
	'medium' : 'point 1',
	'maze_simple' : 'point 3',
	'double_I' : 'point 2',
	'double_L' : 'point 4'
}

diff_to_name_empty = {
	'medium' : ' ',
	'maze_simple' : ' ',
	'double_I' : ' ',
	'double_L' : ' '
}

if not top_title: 
	diff_to_name = diff_to_name_empty
count = 0
plot_count = 0
n_col = 2
subsample = 5
for env in envs:
	#fig.suptitle(env)
	col = 0
	for f, field in enumerate(fields):
		plt.clf()
		fig, ax = plt.subplots(1, 1, figsize=(12, 12))
		plot_list = []
		for s, setting in enumerate(settings):
			#path = os.path.join(dir, env, setting, '*', 'progress.csv')
			path = dir + '/' + env + '/' + setting + '/*/progress.csv'
			paths = glob.glob(path)
			print("Path:", path)
			# print("Paths:", paths)
			min_rows = np.inf

			# get maze name from variant
			debug = os.path.join(os.path.dirname(paths[0]), 'variant.json')
			variant = json.load(open(os.path.join(os.path.dirname(paths[0]), 'variant.json')))
			maze_name = diff_to_name[variant['difficulty']]

			results = []
			final_results = []
			for j, p in enumerate(paths):
				print("p:", p)
				data = pd.read_csv(p, usecols=[field], header=0)

				if len(data) > max_rows:
					data = data[:max_rows]

				res = np.array(data[field], dtype=np.float64)

				if separate:
					if f == 0:
						label = setting  + '-' + str(p)
					else:
						label = None
					mean = running_mean(res, subsample)
					x = list(range(len(mean)))
					ax.plot(x, mean, label=label, color=colors[j], linewidth=LINE_WIDTH)
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

				if f == 0:
					label = setting
				else:
					label = None
				plot_list.append(ax.plot(x, mean, label=label, color=colors[s], linewidth=LINE_WIDTH)[0])
				if n > 1:
					ax.fill_between(x, mean - 2 * (std / np.sqrt(n)),
										 mean + 2 * (std / np.sqrt(n)),
								 alpha=0.2, color=colors[s])
		# ax.set_title(field_to_label[field], fontdict={'fontsize': 7})
		#if field in ['remote_evaluation/Average Returns', 'exploration/Average Returns'] and 'point' in env:
		#	bounds = env_to_bounds[env]
		#	ax.set_ylim(bounds)
		if field in ['trainer/Q Loss'] and 'point' in env:
			ax.set_ylim((0, 20))
		if field in ['trainer/Target Policy Grad',]:
			ax.set_ylim((0, 20))
			if small_scale:
				ax.set_ylim((0, 0.04))
		if field in ['trainer/Critic Target Action Grad']:
			ax.set_ylim((0, 5))
			if small_scale:
				ax.set_ylim((0, 0.04))
		if field in ['trainer/Critic Target Action Grad PTH']:
			ax.set_ylim((0, 2.5))
			if small_scale:	
				ax.set_ylim((0, 0.2))	
		if field in ['remote_evaluation/Num Paths']:
			#y_labels = [6, 7, 8, 9] #, 14, 16, 18, 20] # double_L
			y_labels = [5, 10, 15, 20] #, 14, 16, 18, 20] # maze easy
			ax.set_yticks(y_labels)
			

		# STYLE
		ax.set_ylabel(field_to_label[field], fontsize=AXIS_FONT_SIZE)
		if not ylabel: 
			ax.set_ylabel(' ', fontsize=AXIS_FONT_SIZE)
		ax.set_xlabel('Epochs', fontsize=AXIS_FONT_SIZE)
		if not xlabel: 
			ax.set_xlabel(' ', fontsize=AXIS_FONT_SIZE)
		ax.set_title(maze_name, fontdict={'fontsize':TITLE_SIZE})
		ax.ticklabel_format(style='sci',scilimits=(0,0))
		ax.xaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
		ax.yaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
		ax.tick_params(labelsize=TICKS_FONT_SIZE)
		ax.grid(linestyle=":")
		ax.ticklabel_format(useOffset=False, style='plain')

		# if col // n_col in [0, 2]:
		#	 ax.set_ylim((-6000, -1000))
		col += 1
		plot_count += 1
	#fig.legend(loc='lower center', ncol=max(len(settings)//2, 1))
	#fig.savefig(env + '.png')
		if show:
			plt.show(block=False) # remove block=False if you prefer closing the window manually
		if show:
			input("Press Enter to continue...")
		if save:
			if save_inside: 
				output_name = os.path.join(dir, env, settings[0], settings[0])
			else:
				output_name = os.path.join(dir, env, 'z-plots', os.path.basename(os.path.normpath(env)))
			if separate:
				output_name = output_name + '-plot-sep' + suffix + '.png'
			else:
				output_name = output_name + '-plot' + suffix
			print("output " + output_name)
			field_plot = output_name + field_to_label[field]
			fig.savefig(field_plot + '.png', bbox_inches='tight')
			fig.savefig(field_plot + '.pdf', format='pdf', bbox_inches='tight')

			# lgd = fig.legend(ncol=10, loc='center', frameon=False, fontsize=LEGEND_FONT_SIZE)
			# fig.savefig('legend.pdf',format='pdf',bbox_extra_artists=(lgd,), bbox_inches='tight',)

# alternative names
settings = [
	'SAC',
	'OAC',
	'WAC (ours)',
	#'ME-WAC',
]	

if save_legend:
	legendFig = plt.figure("Legend plot")
	legendFig.legend(plot_list, settings, loc='center', ncol=len(settings))
	output_name_lgd = output_name + '-lgd'
	legendFig.savefig(output_name_lgd + '.png')
	legendFig.savefig(output_name_lgd + '.pdf', format='pdf')

plt.close('all')

