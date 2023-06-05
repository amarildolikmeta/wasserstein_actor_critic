# dev
from utils.core import np_ify, torch_ify
import seaborn as sns
import matplotlib.pyplot as plt
from envs.point import PointEnv
from utils.env_utils import env_producer
from optimistic_exploration import get_optimistic_exploration_action
import pickle
import numpy as np
import os

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
TITLE_SIZE= 45

env = 'data/appendix/DIh_SAC_SR/s1'
env = 'data/appendix/DIh_SAC_SR/s2'
env = 'data/appendix/DIh_OAC_SR/s1'
env = 'data/appendix/DIh_OAC_SR/s2'
env = 'data/appendix/DIh_GSWAC_SR/s1'
env = 'data/appendix/DIh_GSWAC_SR/s2'
env = 'data/appendix/MSh_SAC/s1'
env = 'data/appendix/MSh_SAC/s2'
env = 'data/appendix/MSh_OAC/s1'
env = 'data/appendix/MSh_OAC/s2'
env = 'data/appendix/MSh_GSWACmu/s1'
env = 'data/appendix/MSh_GSWACmu/s2'
# 'data/point/double_I/terminal/gs-oac_/DIh_GSWAC_SR/s1/ch/cum_histo1_10.pkl'
comm_dir = os.path.join(env, 'ch')
epoch = 299

tl = 10

# cum_histo1 = np.zeros(shape=[41, 41], dtype=np.float64)

path1 = os.path.join(comm_dir, "cum_histo1_" + str(epoch) + ".npy")
path2 = os.path.join(comm_dir, "cum_histo2_" + str(epoch) + ".npy")
# file1 = open(path1, "rb")
# file2 = open(path2, "rb")
# cum_histo1 = pickle.load(file1)
# cum_histo2 = pickle.load(file2)
# file1.close()
# file2.close()
cum_histo1 = np.load(path1)
cum_histo2 = np.load(path2)

fig, ax = plt.subplots(1,1,figsize=(12, 12))

cum_histo1 = np.flip(cum_histo1, axis=0)
JG11 = sns.heatmap(data=cum_histo1, vmax=1000, cbar=False, cmap="jet_r", ax=ax)

#ax.set_title('title')
ax.set(xticklabels=[], yticklabels=[])
ax.set_xlabel('x', fontsize=AXIS_FONT_SIZE)
#ax.set_ylabel(f'{hp_names[hp0]}', fontsize=AXIS_FONT_SIZE)
ax.set_ylabel('y', fontsize=AXIS_FONT_SIZE)
# here set the labelsize by 20
ax.tick_params(labelsize=TICKS_FONT_SIZE)

fig.savefig(os.path.join(env, 'cum_histo1_' + str(epoch)))
fig.savefig(os.path.join(env, 'cum_histo1_' + str(epoch) + '.pdf'), format='pdf')
plt.close()

plt.close('all')
plt.style.use('default')
plt.rc('font', family='serif')

fig, ax = plt.subplots(1,1,figsize=(12, 12))

cum_histo2 = np.flip(cum_histo2, axis=0)
JG11 = sns.heatmap(data=cum_histo2, vmax=1000, cbar=False, cmap="jet_r", ax=ax)

ax.set(xticklabels=[], yticklabels=[])
ax.set_xlabel('x', fontsize=AXIS_FONT_SIZE)
#ax.set_ylabel(f'{hp_names[hp0]}', fontsize=AXIS_FONT_SIZE)
ax.set_ylabel('y', fontsize=AXIS_FONT_SIZE)

fig.savefig(os.path.join(env, 'cum_histo2_' + str(epoch)))
fig.savefig(os.path.join(env, 'cum_histo2_' + str(epoch) + '.pdf'), format='pdf')
plt.close()
