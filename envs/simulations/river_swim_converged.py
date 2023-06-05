import os 
import re
import sys
import argparse 
import json
import numpy as np

from os.path import dirname as up
repo_dir = up(up(up(os.path.realpath(__file__))))
sys.path.append(repo_dir)

# import project
from utils.core import np_ify, torch_ify
from utils.pythonplusplus import load_gzip_pickle
from utils.variant_util import build_variant
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from utils.variant_util import build_variant
from envs.river_swim_continuous import RiverSwimContinuous
from trainer.trainer import SACTrainer


# path
#dir = 'data/riverswim/25/001' # has itr for all runs
#dir = './data/riverswim/25/g004-g-oac_'
#dir = './data/riverswim/25/005-poac2L'
#dir = './data/riverswim/25/006-poac2e'
#dir = './data/riverswim/25/007-notb'
#dir = 'data/g008/riverswim/25/g-oac_'
#dir = 'data/riverswim/25/g008-goac-weak'
#dir = 'data/riverswim/25/oac_/009/delta_40_beta_6'
dir = 'data/riverswim/25/gn001'

# save in file
save_output = True

if save_output: 
	orig_stdout = sys.stdout
	f = open(os.path.join(repo_dir, dir, 'converge.txt'), 'w+')
	sys.stdout = f

# handle when multiple runs are in the folder
multiple_runs = True 
if os.path.exists(os.path.join(dir, 'progress.csv')):
	multiple_runs = False

if multiple_runs:
	paths = [ os.path.join(dir, subf) for subf in os.listdir(dir) if os.path.isdir(os.path.join(dir, subf)) ]
else:
	paths = [dir]

# onlykeep the ones with 
paths = [ path for path in paths if any(f.startswith('itr') for f in os.listdir(path))  ]
print('PATHS:')
print(paths)

# get info from variant.json
variant_path = os.path.join(paths[0], 'variant.json')
variant = json.load(open(variant_path, 'r'))
alg = variant['alg']
a_kwargs = variant['algorithm_kwargs']
num_expl_steps_per_train_loop = a_kwargs['num_expl_steps_per_train_loop']
num_train_loops_per_epoch = a_kwargs['num_train_loops_per_epoch']

# generate trainer
trainer = build_variant(variant, return_replay_buffer=False, return_collectors=False)['trainer']

## get files and order them (functions)
def atoi(text):
	return int(text) if text.isdigit() else text

def natural_keys(text):
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

for path in paths:
	## get files and order them
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	r = re.compile("itr.*..zip_pkl")
	itr_files = list(filter(r.match, files))

	itr_files.sort(key=natural_keys)

	# test epochs
	found = False
	for itr in itr_files:
		experiment = os.path.join(path, itr)
		exp = load_gzip_pickle(experiment)
		trainer.restore_from_snapshot(exp['trainer'])

		pure = True
		for s in range(0,26):
			s = np.array([s])
			if alg in ['p-oac', 'g-oac']:
				policy = trainer.target_policy
			elif alg == 'oac':
				policy = trainer.policy
			else:
				raise ValueError('Algorithm not implemented')
			a = policy(
				obs=torch_ify(s), reparameterize=True,
				return_log_prob=True, 
				deterministic=True
			)
			a = np_ify(a[0])
			if a != 1:
				pure = False
		if pure:
			print('\n'+path)
			print("first optimal iteration: " + itr)
			itr_int = natural_keys(itr)[1]
			tot_expl_steps = itr_int * num_expl_steps_per_train_loop * num_train_loops_per_epoch
			print('~expl steps before convergence: ' + str(tot_expl_steps))
			found = True
			break
	if not found:
		print('\n'+path)
		print('Never Optimal')

# reset stdout, close file and print file content
if save_output:  
	sys.stdout = orig_stdout
	f.close()
	with open(os.path.join(dir, 'converge.txt')) as f:
		contents = f.read()
		print(contents)
