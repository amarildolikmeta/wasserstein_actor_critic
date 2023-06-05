# bounds are in the WAC folder google drive
import numpy as np
import os

import json
 
# change working dir to file
abspath = os.path.abspath(__file__)
print(abspath)
parent_abspath = os.path.dirname(abspath)
print(parent_abspath)
os.chdir(parent_abspath)

# get all the files but the python file
files = [f for f in os.listdir(os.getcwd()) if os.path.isfile(os.path.join(os.getcwd(), f))]
files.remove('bounds.py')  

mins = np.empty((376,0), np.double)
maxs = np.empty((376,0), np.double)

# concat maxs and mins
for file in files:
	with open(file) as f:
		a = np.loadtxt(f, usecols=range(1,377))
		mins = np.append(mins, np.expand_dims(a[0], axis=1), axis=1) 
		maxs = np.append(maxs, np.expand_dims(a[1], axis=1), axis=1) 

# get max of maxs and min of mins
abs_mins = np.amin(mins, axis=1)
abs_maxs = np.amax(maxs, axis=1)

final_bounds = np.stack((abs_mins, abs_maxs), axis=0) 

# test how many unused states
## a = (abs_mins == abs_max)

np.save('../humanoid_bounds.npy', final_bounds)

new = np.load('../humanoid_bounds.npy')

