import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns

from utils.variant_util import get_q_producer
from utils.core import np_ify, torch_ify

# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed) 
# random.seed(seed)

# SHARE LAYERS

q_producer = get_q_producer(obs_dim=1, action_dim=1, hidden_sizes=[16] * 1, output_size=2)

log_std = 1.0601317681000446
q = q_producer(bias=np.array([5, log_std]), positive=[False, True], train_bias=True)

min_state, max_state = -1, 1 
#min_state, max_state = 0, 25
min_action, max_action = -1,1

#delta_state = 0.5
delta_state = 0.04
delta_action = 0.05

xs = np.array([min_state + i * delta_state for i in range(int((max_state - min_state) / delta_state + 1))])
ya = np.array([min_action + i * delta_action for i in range(int((max_action - min_action) / delta_action + 1))])

data_heatmap = np.zeros((len(xs), len(ya)))
mean_heatmap = np.zeros((len(xs), len(ya)))

for i in range(len(xs)): # states
	for j in range(len(ya)): # actions
		o = np.array([xs[i]])
		ob = np.array(o).reshape((1, 1))
		a = np.array([ya[j]])
		ac = np.array(a).reshape((1, a.shape[-1]))
		# do what predict does
		ob = torch_ify(ob)
		ac = torch_ify(ac)
		
		qs = q(ob, ac)
		if True: # share_layer
			std = qs[:, 1].unsqueeze(-1)
			mean = qs[:, 0].unsqueeze(-1)
		else:
			pass
			#std = std(ob, ac)
		
		data_heatmap[i, j] = std
		mean_heatmap[i, j] = mean

num_ticks = 9
yticks = np.linspace(0, len(ya) - 1, len(ya), dtype=int)
# the content of labels of these yticks
yticklabels = np.array(["%.1f" % ya[idx] for idx in yticks])
xticks = np.linspace(0, len(xs) - 1, num_ticks, dtype=int)
# the content of labels of these yticks
xticklabels = np.array(["%.1f" % xs[idx] for idx in xticks])
fig = plt.figure(figsize=(13, 8))
	
data_heatmap = data_heatmap.T
mean_heatmap = mean_heatmap.T

JG2 = sns.heatmap(data=mean_heatmap, xticklabels = 2, yticklabels = 8, cmap="jet_r")

JG2.set_title('Std')
#JG2.set_xticklabels(np.arange(0,26))
JG2.set_yticklabels(np.round(np.linspace(-10,10,6))/10)

plt.show()
plt.close(fig)

