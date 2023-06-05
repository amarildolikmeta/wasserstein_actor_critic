

# SAC 

![image](/data/point/maze_simple/maze_simple.png)

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=3
let "seeds++"

lr_s=(1e-4 3e-4 10e-4)
ls_s=(32 64 128 256)

for lr in "${lr_s[@]}"
do
	for ls in "${ls_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
			qsub -v seed=$i,lr=$lr,ls=$ls run.sub
		done # WATCHOUT suffix
	done
done

# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N wac_run
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

source run_tosca.sh $seed $lr $ls > ./data/i001s"${seed}"_lr"${lr}"_ls"${ls}".log 2>&1

# run_tosca.sh # -----------------------------

#!/bin/bash
# chmod +x ex_server.sh

python main.py --seed=$1 --domain=point --terminal --clip_state --difficulty=maze_simple --alg=sac --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=$3 --policy_lr=$2 --qf_lr=$2 --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=i001/lr"${2}"_ls"${3}"
```

![image](/data/point/server/i001/i001-plot-32-64.png)

![image](/data/point/server/i001/i001-plot-128-256.png)

The only run that could get to the end

![image](/data/point/server/i001/lr10e-4_ls256/lr10e-4_ls256-plot-sep.png)

![image](/data/point/server/i001/lr10e-4_ls256/s1/heatmaps/hm_299.png)

![image](/data/point/server/i001/lr10e-4_ls256/s2/heatmaps/hm_299.png)

![image](/data/point/server/i001/lr10e-4_ls256/s3/heatmaps/hm_299.png)

![image](/data/point/server/i001/lr10e-4_ls256/s4/heatmaps/hm_299.png)

the default

![image](/data/point/server/i001/lr3e-4_ls256/lr3e-4_ls256-plot-sep.png)

![image](/data/point/server/i001/lr3e-4_ls256/s1/heatmaps/hm_299.png)

![image](/data/point/server/i001/lr3e-4_ls256/s2/heatmaps/hm_299.png)

![image](/data/point/server/i001/lr3e-4_ls256/s3/heatmaps/hm_299.png)

![image](/data/point/server/i001/lr3e-4_ls256/s4/heatmaps/hm_299.png)

Doesn't get there but never fails too badly

# i002/i003 - OAC

```sh
# run with other delta_s
delta_s=(0.1 0.3 1 3 10 20)

python main.py --seed=$seed --domain=point --terminal --clip_state --difficulty=maze_simple --alg=oac --delta=$delta --beta_UB=4.66 --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=i002/delta"${delta}" > ./data/i002_s"${seed}"_delta"${delta}".log 2>&1
```

![image](/data/point/server/i002/i002-plot2.png)

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=4
let "seeds++"

delta_s=(14 18 22 26)

for delta in "${delta_s[@]}"
do
	for ((i=1;i<seeds;i+=1))
	do
		qsub -v seed=$i,delta=$delta run.sub
	done # WATCHOUT suffix
done

# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N wac_run
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --seed=$seed --domain=point --terminal --clip_state --difficulty=maze_simple --alg=oac --delta=$delta --beta_UB=4.66 --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i003/delta"${delta}" > ./data/i003_s"${seed}"_delta"${delta}".log 2>&1
```

![image](/data/point/server/i003/i003-plot2.png)

![image](/data/point/server/i003/delta18/delta18-plot-sep2.png)

![image](/data/point/server/i003/delta18/s1/heatmaps/hm_299.png)

![image](/data/point/server/i003/delta18/s2/heatmaps/hm_299.png)

![image](/data/point/server/i003/delta18/s3/heatmaps/hm_299.png)

![image](/data/point/server/i003/delta18/s4/heatmaps/hm_299.png)

# GOAC

```sh
delta_s=(0.8 0.9 0.95)
r_max_s=(0.1 0.3 0.5)
lr_s=(1e-4 3e-4 10e-4)

for delta in "${delta_s[@]}"
do
	for r_max in "${r_max_s[@]}"
	do
		for lr in "${lr_s[@]}" 
		do
			for ((i=1;i<seeds;i+=1))
			do
				r_min=$(expr -$r_max)
				python main.py --difficulty=maze_simple --seed=$i --domain=point --terminal --clip_state --alg=g-oac --delta=$delta --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --qf_lr=$lr --std_lr=$lr --policy_lr=$lr --expl_policy_lr=$lr --r_min=$r_min --r_max=$r_max --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i004/delta-"${delta}"_r-"${r_max}"_lr"${lr}" &
			done 
		done
	done
done
```

![image](/data/point/server/i004/i004-plot-0.8.png)

![image](/data/point/server/i004/i004-plot-0.9.png)

![image](/data/point/server/i004/i004-plot-0.95.png)

The most promising results seem to be 

* `delta=0.9`
* `lr>3e-4`

The best one seem to be the most "default": `delta-0.9_r-0.5_lr3e-4`

![image](/data/point/server/i004/delta-0.9_r-0.5_lr3e-4/delta-0.9_r-0.5_lr3e-4-plot-sep-0.95.png)

![image](/data/point/server/i004/delta-0.9_r-0.5_lr3e-4/s1/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.5_lr3e-4/s2/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.5_lr3e-4/s3/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.5_lr3e-4/s4/heatmaps/hm_299.png)

-------

Another good one is `delta-0.9_r-0.1_lr10e-4`

![image](/data/point/server/i004/delta-0.9_r-0.1_lr10e-4/delta-0.9_r-0.1_lr10e-4-plot-sep-0.95.png)

![image](/data/point/server/i004/delta-0.9_r-0.1_lr10e-4/s1/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.1_lr10e-4/s2/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.1_lr10e-4/s3/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.1_lr10e-4/s4/heatmaps/hm_299.png)


------

`delta-0.9_r-0.3_lr10e-4`

two of them went really bad


![image](/data/point/server/i004/delta-0.9_r-0.3_lr10e-4/s1/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.3_lr10e-4/s2/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.3_lr10e-4/s3/heatmaps/hm_299.png)

![image](/data/point/server/i004/delta-0.9_r-0.3_lr10e-4/s4/heatmaps/hm_251.png)

# i010 - let's tune the qty on this last setting

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=4
let "seeds++"

qty_s=(0.3 0.7 1)
weight_s=(0.5 1)

for qty in "${qty_s[@]}"
do
	for weight in "${weight_s[@]}" 
	do
		for ((i=1;i<seeds;i+=1))
		do
			qsub -v seed=$i,prv_std_qty=$qty,prv_std_weight=$weight run.sub &
		done 
	done
done

# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N wac_run
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --difficulty=maze_simple --seed=$i --domain=point --terminal --clip_state --alg=g-oac --delta=0.9 --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --qf_lr=10e-4 --std_lr=10e-4 --policy_lr=10e-4 --expl_policy_lr=10e-4 --r_min=-0.3 --r_max=0.3 --prv_std_qty=$qty --dont_use_target_std --prv_std_weight=$weight --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i009/qty"${qty}"_weight"${weight}" > ./data/i009_s"${seed}"_qty"${qty}"_weight"${weight}".log 2>&1
```

![image](/data/point/server/i010/i010-plot_good.png)


# i013 - ours tuning also qty

```sh
r_max_s=(0.3 0.5)
lr_s=(3e-4 10e-4)
qty_s=(0 0.1 0.3 0.5 1)

for qty in "${qty_s[@]}"
do
	for r_max in "${r_max_s[@]}"
	do
		for lr in "${lr_s[@]}" 
		do
			for ((i=1;i<seeds;i+=1))
			do
				r_min=$(expr -$r_max)
				python main.py --difficulty=maze_simple --seed=$i --domain=point --terminal --clip_state --alg=g-oac --delta=0.9 --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --qf_lr=$lr --std_lr=$lr --policy_lr=$lr --expl_policy_lr=$lr --r_min=$r_min --r_max=$r_max --prv_std_qty=$qty --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i013/qty-"${qty}"_r-"${r_max}"_lr"${lr}" &
			done 
		done
	done
done
```

![image](/data/point/server/i013/i013-plot_0.3.png)

![image](/data/point/server/i013/i013-plot_0.5.png)

![image](/data/point/server/i013/qty-0.3_r-0.5_lr10e-4/s1/heatmaps/hm_299.png)

![image](/data/point/server/i013/qty-0.3_r-0.5_lr10e-4/s2/heatmaps/hm_299.png)

![image](/data/point/server/i013/qty-0.3_r-0.5_lr10e-4/s3/heatmaps/hm_299.png)

# i014 - SAC - double_L

![image](/data/point/double_L/double_L.png)

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=5
let "seeds++"

for ((i=1;i<seeds;i+=1))
do
	qsub -v seed=$i run.sub
done # WATCHOUT suffix


# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N i014
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --seed=$seed --domain=point --terminal --clip_state --difficulty=double_L --alg=sac --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i014 > ./data/i014s"${seed}".log 2>&1
```

![image](/data/point/server/i014/i014-plot-sep.png)

![image](/data/point/server/i014/s1/heatmaps/hm_299.png)

![image](/data/point/server/i014/s2/heatmaps/hm_299.png)

![image](/data/point/server/i014/s3/heatmaps/hm_299.png)

![image](/data/point/server/i014/s4/heatmaps/hm_299.png)

# i015 - OAC Leaky - double_L

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=5
let "seeds++"

for ((i=1;i<seeds;i+=1))
do
	qsub -v seed=$i run.sub
done # WATCHOUT suffix

# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N i015
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --seed=$seed --domain=point --terminal --clip_state --difficulty=double_L --alg=oac --delta=18 --beta_UB=4.66 --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i015 > ./data/i015_s"${seed}".log 2>&1
```

![image](/data/point/server/i015/i015-plot-sep.png)

![image](/data/point/server/i015/s1/heatmaps/hm_299.png)

![image](/data/point/server/i015/s2/heatmaps/hm_299.png)

![image](/data/point/server/i015/s3/heatmaps/hm_299.png)

![image](/data/point/server/i015/s4/heatmaps/hm_299.png)

![image](/data/point/server/i015/s5/heatmaps/hm_299.png)

# i016t - ours tuning also qty

```sh
rbs_s=(1e4 1e5 1e6)
lr_s=(0.3e-3 1e-3 3e-3)
qty_s=(0.1 0.3 0.5)

for qty in "${qty_s[@]}"
do
	for rbs in "${rbs_s[@]}"
	do
		for lr in "${lr_s[@]}" 
		do
			for ((i=1;i<seeds;i+=1))
			do
				python main.py --difficulty=double_L --seed=$i --domain=point --terminal --clip_state --alg=g-oac --delta=0.9 --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --qf_lr=$lr --std_lr=$lr --policy_lr=$lr --expl_policy_lr=$lr --r_min=-0.5 --r_max=0.5 --prv_std_qty=$qty --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=$rbs --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i016t/qty"${qty}"_rbs"${rbs}"_lr"${lr}" &
			done 
		done
	done
done
```

![image](/data/point/server/i016t/i016t-plot_0.3.png)

![image](/data/point/server/i016t/qty0.3_rbs1e5_lr0.3e-3/qty0.3_rbs1e5_lr0.3e-3-plot-sep.png)

![image](/data/point/server/i016t/qty0.3_rbs1e5_lr0.3e-3/s1/heatmaps/hm_299.png)

![image](/data/point/server/i016t/qty0.3_rbs1e5_lr0.3e-3/s2/heatmaps/hm_290.png)

![image](/data/point/server/i016t/qty0.3_rbs1e5_lr0.3e-3/s3/heatmaps/hm_299.png)

![image](/data/point/server/i016t/qty0.3_rbs1e5_lr0.3e-3/s4/heatmaps/hm_299.png)

# i017 - SAC - maze_med

![image](/data/point/maze_med/maze_med.png)

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=5
let "seeds++"

for ((i=1;i<seeds;i+=1))
do
	qsub -v seed=$i run.sub
done # WATCHOUT suffix


# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N i017
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --seed=$seed --domain=point --terminal --clip_state --difficulty=maze_med --alg=sac --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i017 > ./data/i017s"${seed}".log 2>&1
```

![image](/data/point/server/i017/i017-plot-sep.png)

![image](/data/point/server/i017/s1/heatmaps/hm_299.png)

![image](/data/point/server/i017/s2/heatmaps/hm_299.png)

![image](/data/point/server/i017/s3/heatmaps/hm_299.png)

![image](/data/point/server/i017/s4/heatmaps/hm_299.png)

![image](/data/point/server/i017/s5/heatmaps/hm_299.png)

# i018 - OAC - maze_med

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=5
let "seeds++"

for ((i=1;i<seeds;i+=1))
do
	qsub -v seed=$i run.sub
done # WATCHOUT suffix

# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N i018
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --seed=$seed --domain=point --terminal --clip_state --difficulty=maze_med --alg=oac --delta=18 --beta_UB=4.66 --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i018 > ./data/i018_s"${seed}".log 2>&1
```

![image](/data/point/server/i018/i018-plot-sep.png)

![image](/data/point/server/i018/s1/heatmaps/hm_299.png)

![image](/data/point/server/i018/s2/heatmaps/hm_299.png)

![image](/data/point/server/i018/s3/heatmaps/hm_299.png)

![image](/data/point/server/i018/s4/heatmaps/hm_299.png)

![image](/data/point/server/i018/s5/heatmaps/hm_299.png)

# i019 - ours ?? - maze_med

```sh
# run.sh # -----------------------------------

#!/bin/bash

seeds=5
let "seeds++"

for ((i=1;i<seeds;i+=1))
do
	qsub -v seed=$i run.sub &
done 

# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N i019
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --difficulty=maze_med --seed=$seed --domain=point --terminal --clip_state --alg=g-oac --delta=18 --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --qf_lr=1e-3 --std_lr=1e-3 --policy_lr=1e-3 --expl_policy_lr=1e-3 --r_min=-0.5 --r_max=0.5 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i019 > ./data/i019s"${seed}".log 2>&1
```

![image](/data/point/server/i019/i019-plot-sep.png)

![image](/data/point/server/i019/s1/heatmaps/hm_299.png)

![image](/data/point/server/i019/s2/heatmaps/hm_299.png)

![image](/data/point/server/i019/s3/heatmaps/hm_299.png)

![image](/data/point/server/i019/s4/heatmaps/hm_299.png)

![image](/data/point/server/i019/s5/heatmaps/hm_299.png)

# comparison

![image](/data/point/point-plot_maze_simple_comp.png)

![image](/data/point/point-plot_double_L_comp.png)

![image](/data/point/point-plot_maze_med_comp.png)

# even in the best runs, especially at the beginning the policy get stuck

I think we have to work on cliff and then we have to identify stats that we can observe in higher dimension so that if the problem still occurs in higher dimensions we can address it.

## Stuck for over 10 epochss

![image](/data/point/server/i013/qty-0.3_r-0.5_lr10e-4/s1/heatmaps/hm_1.png)

![image](/data/point/server/i013/qty-0.3_r-0.5_lr10e-4/s1/heatmaps/hm_12.png)

## Stuck for 8

![image](/data/point/server/i013/qty-0.3_r-0.5_lr10e-4/s4/heatmaps/hm_0.png)

![image](/data/point/server/i013/qty-0.3_r-0.5_lr10e-4/s4/heatmaps/hm_9.png)
