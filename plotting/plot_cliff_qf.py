import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from envs.cliff_continuous_mono import CliffWorldContinuousMono


mdp = CliffWorldContinuousMono(dim=(12,),sigma_noise=0)

delta_state = 0.2
xs = np.arange(0,121,delta_state*10)/10
delta_action = 0.05
ya = np.round(np.linspace(-100,100,int(delta_action*1000)+1))/100 # (-1,+1,0.04)

def opt_policy_noise(s):
	if s <= 3.75:
		return 1
	if 3.75 < s < 4.75:
		return -s + 4.75
	if 4.75 <= s < 5.5:
		return 1 
	if 5 <= s <= 5.5: # cliff
		pass
	if s > 5.5:
		return 1

def opt_policy_det(s):
	if 4 <= s <= 4.5:
		return - s + 5 - 0.001
	return 1

qf = np.zeros((len(xs), len(ya)))
ps = np.zeros(len(xs))

for s0 in range(len(xs)): # states
	ps[s0] = opt_policy_det(xs[s0])
	for a0 in range(len(ya)): # actions  
		if xs[s0] == 12 and ya[a0] == 1:
			debug = 0
		s = mdp.reset(state=np.array([xs[s0]]))
		s, r, done, _ = mdp.step(ya[a0])
		tot_r = r
		t = 1
		while not done: 
			a = opt_policy_det(s[0])
			s, r, done, _ = mdp.step(a)
			tot_r += r * (mdp.gamma ** t) 
			t += 1
		qf[s0, a0] = tot_r

np.save('./data/cliff_mono/opt_qf.npy', qf)
np.save('./data/cliff_mono/opt_ps_det.npy', ps)

qf = qf.T
max_state = 12
max_action, min_action = +1, -1
# 60 -> 0,12
# I want 12 tick -> 60/12=5
xtl = int((qf.shape[1] - 1) / max_state - 0)
# 50 -> -1,+1
# I want one every 0.2
# 2/0.2 = 10
# 50/10 = 5
d_y = 0.5
ytl = int((qf.shape[0] - 1) / ((max_action - min_action) / d_y))
fig = plt.plot(figsize=(20,8))
ax = sns.heatmap(qf, xticklabels = xtl, yticklabels = ytl, cmap="jet_r")
ax.set_xticklabels(np.arange(0,max_state+1))
ax.set_yticklabels(np.round(np.linspace(-10,10,int((max_action - min_action)/d_y+1)))/10)
plt.show()


