import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import glob
import scipy.stats as sts
import ast
from scipy.interpolate import interp1d
from collections import OrderedDict
from matplotlib.transforms import blended_transform_factory
import matplotlib.lines as mlines
plt.close('all')
plt.style.use('default')
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

LEGEND_FONT_SIZE = 28
AXIS_FONT_SIZE = 28
TICKS_FONT_SIZE = 26
MARKER_SIZE = 10
LINE_WIDTH = 3.0
TITLE_SIZE= 28

figs = []
# %matplotlib inline
def running_mean(x, N):
    divider = np.convolve(np.ones_like(x), np.ones((N,)), mode='same')
    return np.convolve(x, np.ones((N,)), mode='same') / divider


linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),

     ('loosely dashdashdotted', (0, (3, 10, 3, 10, 1, 10))),
     ('dashdashdotted',         (0, (3, 5, 3, 5, 1, 5))),
     ('densely dashdashdotted', (0, (3, 1, 3, 1, 1, 1)))])

ls = [linestyles['solid'], linestyles['dashed'], linestyles['dashdashdotted'],linestyles['densely dotted'],
      linestyles['dotted'], linestyles['dashdotted'], linestyles['dashdotdotted'], 
      linestyles['densely dashed'],  linestyles['loosely dashdashdotted']]

col = ['c', 'k','orange', 'purple','r', 'b', 'g', 'y','brown','magenta','#BC8D0B',"#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^','2','1','3','4']

def plot_ci(xs, ys, conf, n, xlabel=None, ylabel=None, label=None, ax=None, **kwargs):
    
    if np.array(xs).ndim > 1:
        all_x = np.array(list(set(np.concatenate(xs).ravel().tolist())))
        all_x = np.sort(all_x)
        last_x = min(map(lambda x: max(x), xs))
        first_x = max(map(lambda x: min(x), xs))
        all_ys = []

        pred_x = np.linspace(first_x, last_x, 200)

        for i in range(len(xs)):
            f = interp1d(xs[i], ys[i], "linear")
            all_ys.append(f(pred_x))
        all_ys = np.array(all_ys)
    else:
        all_x = xs
        all_ys = ys
        pred_x = xs
    
    
    N = 4
    for i in range(all_ys.shape[0]):
        all_ys[i] = running_mean(all_ys[i], N)
    
    data_mean = np.mean(all_ys, axis=0)
    data_std = np.std(all_ys, axis=0) + 1e-24
    interval = sts.t.interval(conf, n-1,loc=data_mean,scale=data_std/np.sqrt(n))
    col = kwargs.get('color')
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        figs.append(fig)
    
   #y = running_mean(data_mean, N)
    
    if label is None:
        ax.plot(pred_x, data_mean, **kwargs)
    else:
        ax.plot(pred_x, data_mean, label=label, **kwargs)
        
    if col is not None:
        ax.fill_between(pred_x, interval[0], interval[1], alpha=0.2, color=col, linewidth=0.)
    else:
        ax.fill_between(pred_x, interval[0], interval[1], alpha=0.2, linewidth=0.)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
    ax.ticklabel_format(style='sci',scilimits=(0,0))
    ax.xaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
    ax.yaxis.offsetText.set_fontsize(TICKS_FONT_SIZE+2)
    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    
    return ax

environments = ['Taxi','Chain','RiverSwim','SixArms']
algorithms = ['particle-ql','gaussian-ql','boot-ql','ql','delayed-ql','mbie']
policies = [['weighted','ucb'],['weighted-gaussian','ucb'],['boot','weighted'],['eps-greedy', 'boltzmann'],[''],['rmax']]
updates = [['weighted','mean'],['weighted','mean'],[''], [''],['']]
alg_to_label={
        'particle-ql':'P-WQL',
        'ql':'QL',
        'boot-ql':'BQL',
        'gaussian-ql':'G-WQL',
        'r-max':'Rmax',
        'mbie':'MBIE',
        'delayed-ql':'Delayed-QL'
}
policy_to_label={
        'weighted':'PS',
        'vpi':'V',
        'eps-greedy':'e-Gree',
        'boltzmann':'Boltz',
        'boot':'Boot',
        'weighted-gaussian':'PS',
        'rmax':'',
        'mbie':'',
        'ucb':'OFU',
        '':''
}
update_to_label={
        'weighted':'PO',
        'mean':'MO',
        'optimistic':'OO',
        '':''
}
alg_to_double_vec ={
        'particle-ql':['False'],
        'ql':['True'],
        'boot-ql':['True'],
        'gaussian-ql':['False'],
        'r-max':['False'],
        'mbie':['False'],
        'delayed-ql':['False']
}
env_to_algs= {
    'Taxi':(['gaussian-ql', 'particle-ql', 'boot-ql', 'ql','delayed-ql'], 
            [['weighted-gaussian',],['weighted',],['boot'], ['eps-greedy'],['']],
            [['mean'],['mean'],[''],[''],['']]),
    'Gridworld':(['gaussian-ql','particle-ql'],
            [['weighted-gaussian','ucb'],['weighted','ucb']],
            [ ['weighted','mean','optimistic'],['weighted','mean','optimistic']]),
    'Chain':(['gaussian-ql', 'particle-ql', 'boot-ql', 'ql','delayed-ql','mbie'], 
            [['ucb',],['ucb',],['boot'], ['eps-greedy'],[''],['mbie']],
            [['mean'],['mean'],[''],[''],[''],['']]),
    'RiverSwim':(['gaussian-ql', 'particle-ql', 'boot-ql', 'ql','delayed-ql','mbie'], 
            [['weighted-gaussian',],['weighted',],['boot'], ['eps-greedy'],[''],['mbie']],
            [['weighted'],['weighted'],[''],[''],[''],['']]),
    'SixArms':(['gaussian-ql', 'particle-ql', 'boot-ql', 'ql','delayed-ql','mbie'], 
            [['weighted-gaussian',],['weighted',],['boot'], ['eps-greedy'],[''],['mbie']],
            [['weighted'],['weighted'],[''],[''],[''],['']]),
    'KnightQuest':(['gaussian-ql', 'particle-ql', 'boot-ql', 'ql','delayed-ql'], 
            [['weighted-gaussian',],['weighted',],['boot'], ['eps-greedy'],['']],
            [['weighted'],['weighted'],[''],[''],['']]),
    'Loop':(['gaussian-ql', 'particle-ql', 'boot-ql', 'ql','delayed-ql'], 
            [['weighted-gaussian',],['weighted',],['boot'], ['eps-greedy'],['']],
            [['weighted'],['weighted'],[''],[''],['']]),
    
}
conf = 0.95
subsample = 10
legend_labels=[]
legend_colors=[]
legend_markers=[]
n_col = 2
limits=[(-50,0),(-10000,80000)]
for e,env in enumerate(environments):
    if e%n_col == 0:
        fig, ax = plt.subplots(1, n_col, figsize=(12, 5))
    #fig.suptitle(env)
    algs, alg_policies, alg_updates = env_to_algs[env]
    i = 0    
    for j, alg in enumerate(algs):
        double_vec = alg_to_double_vec[alg]
        
        for pi in alg_policies[j]:
            for u in alg_updates[j]:
                for double in double_vec:
                    if pi in ['ucb']:
                        paths = glob.glob("./tabular_data/" + env + "/" + alg + "/results_" + pi + "_*_"+ u + "_*_"+"double=False_*.npy")
                    elif alg in ['delayed-ql']:
                        #results_delayed_m=1900_1547483900.8137288.npy
                        paths = glob.glob("./tabular_data/" + env + "/" + alg + "/results_delayed_*.npy")
                    elif alg not in ['gaussian-ql']:
                        paths = glob.glob("./tabular_data/" + env + "/" + alg + "/results_" + pi + "_*_"+ u + "_*_"+"double="+ double+"*.npy")
                    else:
                        paths = glob.glob("./tabular_data/" + env + "/" + alg + "/results_" + pi + "_*_"+ u + "_*_"+"double="+ double+"*_log_lr=False"+ "*.npy")                  
                    
                    #print(paths)
                    for p in paths:
                        results = np.load(p)
                        #print(results.shape)
                        n_runs = results.shape[0]
                        try:
                            timesteps = np.cumsum(results[:, 0, :, 0][0])
                            cum_reward_train = results[:, 0, :, 3]
                            cum_reward_test = results[:, 1, :, 3]
                        except:
                            b = np.zeros(shape=results.shape+(11,))
                            for  (x,h,k),value in np.ndenumerate(results):
                                for l in range(11):
                                    b[x,h,k,l]=results[x,h,k][l]
                            results=b
                            timesteps = np.cumsum(results[:, 0, :, 0][0])
                            cum_reward_train = results[:, 0, :, 3]
                            cum_reward_test = results[:, 1, :, 3]
                        

                        #print(cum_reward_test)
                        '''if alg in ['gaussian-ql','particle-ql']:
                            lab = alg_to_label[alg] + "-" +policy_to_label[pi]+ "-" + update_to_label[u]
                            
                        elif alg in ['boot-ql','ql','delayed-ql']:
                            lab = alg_to_label[alg]
                        else:
                            raise ValueError()''' #alg_to_label[alg] + "-" + policy_to_label[pi] + "-" + update_to_label[u]
                        ax[e%n_col].set_title(env, fontdict={'fontsize':20})
                        #ax[0].set_title('Online', fontdict={'fontsize':TITLE_SIZE})
                        indexes = [i*subsample for i in range(len(timesteps) // subsample)]
                        if e==1:
                            lab = alg_to_label[alg] #+ '-' +policy_to_label[pi] + '-' + update_to_label[u]
                            legend_labels.append(lab)
                            legend_colors.append(col[i])
                            legend_markers.append(markers[i])
                        if e%n_col ==0:
                            y_lab ='Average return'
                        else:
                            y_lab = None
                            
                        #y_lab = 'Average return'
                        lab = ''
                        
                        ax[e%n_col] = plot_ci(timesteps[indexes],cum_reward_train[:,indexes], conf, 
                                    n_runs, xlabel='Samples', ylabel=y_lab, label=lab,
                                    ax=ax[e%n_col], linewidth=LINE_WIDTH, linestyle=ls[0], color=col[i], marker=markers[i], markersize=MARKER_SIZE)
                        #ax[0].set_ylim(limits[e])
                        #ax[1].set_title('Offline', fontdict={'fontsize':TITLE_SIZE})
                        #ax[1] = plot_ci(timesteps[indexes], cum_reward_test[:,indexes], conf, 
                        #            n_runs, xlabel='Samples', 
                        #            ax=ax[1], linewidth=LINE_WIDTH, linestyle=ls[0], color=col[i], marker=markers[i], markersize=MARKER_SIZE)
                        #ax[1].set_ylim(limits[e])'''
                        ax[e%n_col].grid(linestyle=":")
                        ax[e%n_col].locator_params(axis='x', nbins=6) 
                        #ax[1].grid(linestyle=":")
                        #ax[1].locator_params(axis='x', nbins=6) 
                        i += 1
    
        
    #lgd=fig.legend(loc='lower center', 
     #                     ncol=5, fancybox=True, shadow=True,bbox_to_anchor=(0.45, -0.015))
    '''if e%n_col == n_col-1:  
        e1 = environments[e-1]
        fig.savefig('images/appendix-'+e1+"-"+env+".pdf",format='pdf', bbox_inches='tight')'''
    fig.savefig('images2/'+env+".pdf",format='pdf', bbox_inches='tight')
    

fig = plt.figure(figsize=(7, 0.2))
patches = [
    mpatches.Patch(color=legend_colors[i], label=legend_labels[i], hatch=legend_markers[0])
    for i in range(len(legend_colors))]
patches = [
	mlines.Line2D(
		[], [], color=legend_colors[i], marker=legend_markers[i], markersize=MARKER_SIZE,
    	label=legend_labels[i], linewidth=LINE_WIDTH
	) for i in range(len(legend_labels)) 
]

lgd=fig.legend(patches, legend_labels, ncol=10, loc='center', frameon=False, fontsize=LEGEND_FONT_SIZE)
fig.savefig('images2/legend.pdf',format='pdf',bbox_extra_artists=(lgd,), bbox_inches='tight',)



