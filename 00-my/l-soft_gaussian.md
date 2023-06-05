
# code 

```py
target_q_values = self.target_qf1(next_obs, new_next_actions) - target_alpha * new_log_pi
```


```py
q_new_target_actions = self.qf1(obs, new_target_obs_actions)

target_policy_loss = (target_alpha * target_log_pi - q_new_target_actions).mean()

stds = self.std(obs, new_obs_actions)
upper_bound = q_new_actions + self.standard_bound * stds
policy_loss = (alpha * log_pi - upper_bound).mean()
```

now if `delta=0` or `std=0` (through `--prv_std_qty=0.3` or `--r_min=r_max`)

now we can solve cliff. 

# SAC - riverswim

having no reward SAC cannot solve riverswim non determinisitc

![image](/data/riverswim/25/sac_/x09b/s1/heatmaps/hm_49.png)

# OAC...

```sh
delta_s=(10 18 20 25 15)

for delta in "${delta_s[@]}" 
do
	for ((i=1;i<seeds;i+=1))
	do
		python main.py --seed=$i --domain=riverswim --alg=oac --beta_UB=4.66 --delta=$delta --max_path_length=50 --layer_size=128 --snapshot_mode=none --save_heatmap --save_sampled_data --epochs=20 --no_gpu ---suffix=x12/delta-"${delta}" &
	done
done
```

```sh
# run.sh # -----------------------------------

#!/bin/bash

seed_num=4
let "seed_num++"

delta_s=(10 18 20 25 15)

for delta in "${delta_s[@]}" 
do
	for ((seed=1;i<seed_num;i+=1))
	do
		qsub -v seed=$seed,delta=$delta run.sub 
	done
done


# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N x12
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --seed=$seed --domain=riverswim --alg=oac --beta_UB=4.66 --delta=$delta --max_path_length=150 --layer_size=128 --snapshot_mode=none --save_heatmap --save_sampled_data --epochs=200 --no_gpu --suffix=x12/delta"${delta}" > ./data/x12_s"${seed}"_delta"${delta}".log 2>&1
```

# gs-oac

```sh
for ((i=1;i<seeds;i+=1))
do
  python main.py --seed=$i --domain=riverswim --alg=gs-oac --delta=0.9 --mean_update --max_path_length=50 --layer_size=128 --std_lr=3e-3 --r_min=-0.2 --r_max=0.2 --prv_std_qty=0.1 --dont_use_target_std --snapshot_mode=none --save_heatmap --save_sampled_data --epochs=20 --no_gpu --suffix=y01 &
done
```

![image](/data/riverswim/25/mean_update_/gs-oac_/y01/s1/heatmaps/hm_18.png)

# grid search

* delta=0.9
* mean_update False
* std_lr=3e-4
* lr=3e-4 # all other networks
* layer_size=128
* num_layers=2
* std init (--r_min=-0.2 --r_max=0.2)
* prv_std_qty=0.1
* replay_buffer_size=1e-6
* batch_size=256
* num_expl_steps_per_train_loop=1000
* num_trains_per_train_loop=1000

io forse farei 

lr_s=(3e-4 1e-3) # std and all others
layer_size_s=(128 64)
prv_std_qty_s=(0.05 0.1 0.3) 


```sh
lr_s=(3e-4 1e-3) # std and all others
ls_s=(128 64)
qty_s=(0.05 0.1 0.3) 


for lr in "${lr_s[@]}"
do
	for ls in "${ls_s[@]}" # WATCHOUT suffix
	do
		for qty in "${qty_s[@]}" # WATCHOUT suffix
		do
			for ((i=1;i<seeds;i+=1))
			do
			python main.py --seed=$i --domain=riverswim --alg=gs-oac --delta=0.9 --mean_update --max_path_length=50 --layer_size=$ls --std_lr=$lr --policy_lr=$lr --qf_lr=$lr --r_min=-0.2 --r_max=0.2 --prv_std_qty=$qty --dont_use_target_std --snapshot_mode=none --save_heatmap --save_sampled_data --epochs=20 --no_gpu --suffix=y04/lr"${lr}"_ls"${ls}"_qty"${qty}" &
			done
		done
	done
done
```





