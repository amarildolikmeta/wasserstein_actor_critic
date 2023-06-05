
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

![image](/data/point/i013/i013-plot_0.3.png)

![image](/data/point/i013/i013-plot_0.5.png)

![image](/data/point/i013/qty-0.3_r-0.5_lr10e-4/qty-0.3_r-0.5_lr10e-4-plot-sep.png)

![image](/data/point/i013/qty-0.3_r-0.5_lr10e-4/s1/heatmaps/hm_299.png)

![image](/data/point/i013/qty-0.3_r-0.5_lr10e-4/s2/heatmaps/hm_299.png)

![image](/data/point/i013/qty-0.3_r-0.5_lr10e-4/s3/heatmaps/hm_299.png)

![image](/data/point/i013/qty-0.3_r-0.5_lr10e-4/s4/heatmaps/hm_299.png)

# i016 - goac double_L

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
#PBS -N i011
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --difficulty=double_L --seed=$seed --domain=point --terminal --clip_state --alg=g-oac --delta=0.9 --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --qf_lr=1e-3 --std_lr=1e-3 --policy_lr=1e-3 --expl_policy_lr=1e-3 --r_min=-0.5 --r_max=0.5 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i016 > ./data/i016s"${seed}".log 2>&1
```

![image](/data/point/i016/i016-plot-sep.png)

Red same policy forever

![image](/data/point/i016/s1/heatmaps/hm_298.png)

Black

![image](/data/point/i016/s2/heatmaps/hm_298.png)

Yellow

![image](/data/point/i016/s3/heatmaps/hm_298.png)

Purple 

![image](/data/point/i016/s4/heatmaps/hm_298.png)


Blue

![image](/data/point/i016/s5/heatmaps/hm_298.png)

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

python main.py --difficulty=maze_med --seed=$seed --domain=point --terminal --clip_state --alg=g-oac --delta=0.9 --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --qf_lr=1e-3 --std_lr=1e-3 --policy_lr=1e-3 --expl_policy_lr=1e-3 --r_min=-0.5 --r_max=0.5 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e6 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=i019 > ./data/i019s"${seed}".log 2>&1
```

![image](/data/point/i019/i019-plot-sep.png)

Red

![image](/data/point/i019/s1/heatmaps/hm_298.png)

Black

![image](/data/point/i019/s2/heatmaps/hm_298.png)

Yellow

![image](/data/point/i019/s3/heatmaps/hm_298.png)

Purple 

![image](/data/point/i019/s4/heatmaps/hm_298.png)

Blue

![image](/data/point/i019/s5/heatmaps/hm_298.png)


