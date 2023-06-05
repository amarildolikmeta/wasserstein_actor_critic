
# SAC - tuning on layers size

![image](/data/point/maze_simple/maze_simple.png)

# WRONG non episodic

Higher layers sizes show significantly better results, this is a problem if we want to run on turing.

Also as soon as I scaled the reward SAC behavior changed significantly, SAC paper says it is an importan hp.

```sh
lr_s=(1e-4 3e-4 10e-4)
ls_s=(32 64 128 256)

for lr in "${lr_s[@]}"
do
	for ls in "${ls_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
			taskset -ca 22-65 python main.py --seed=$i --domain=point --terminal --clip_state --difficulty=maze_simple --alg=sac --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=$ls --policy_lr=$lr --qf_lr=$lr --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=f003/lr"${lr}"_ls"${ls}" &
		done # WATCHOUT suffix
	done
done
```

![image](/data/point/f003/f003-plot.png)

![image](/data/point/f003/f003-plot-lsh.png)

![image](/data/point/f003/f003-plot-lsl.png)

One of the best one, in term of compromise between exploration and convergence to good results is the default `lr=3e-4, ls=256`

![image](/data/point/f003/lr3e-4_ls256/lr3e-4_ls256-plot-sep.png)

![image](/data/point/f003/lr3e-4_ls256/s1/heatmaps/hm_299.png)

![image](/data/point/f003/lr3e-4_ls256/s2/heatmaps/hm_299.png)

![image](/data/point/f003/lr3e-4_ls256/s3/heatmaps/hm_299.png)

## 32

with fewer neurons the algorithm tends to explore much more, to the point where it does enter inside the hole but it can't adapat the q function.

It actually has problem in fitting the much easier local optimum that the other ones find.

![image](/data/point/f003/lr1e-4_ls32/s1/heatmaps/hm_299.png)

![image](/data/point/f003/lr1e-4_ls32/s2/heatmaps/hm_299.png)

![image](/data/point/f003/lr1e-4_ls32/s3/heatmaps/hm_299.png)

Some of these runs enter the whole but just briefly thanks to high entropy

# h001 - search OAC

```sh
delta_s=(14 18 23 27)
beta_s=(3 4 5 6)

for delta in "${delta_s[@]}"
do
	for beta in "${beta_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
			taskset -ca 22-65 python main.py --seed=$i --domain=point --terminal --clip_state --difficulty=maze_simple --alg=oac --delta=$delta --beta_UB=$beta --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=h001/delta"${delta}"_beta"${beta}" &
		done # WATCHOUT suffix
	done
done
```

![image](/data/point/h001/h001-plot-14-18.png)

![image](/data/point/h001/h001-plot-23-27.png)

The best run is in the purple line, which is a set of parameters that shows high variance in the results

![image](/data/point/h001/delta23_beta6/delta23_beta6-plot-sep.png)

![image](/data/point/h001/delta18_beta5/s3/heatmaps/hm_299.png)

One problem here is that what seem to be the right parametes might simply be due to case


# h002 - search goac

![image](/data/point/maze_easy/nose.png)

Wrong maze
```sh
delta_s=(0.6 0.75 0.9)
r_max_s=(0.01 0.03 0.1 0.3)

for r_max in "${r_max_s[@]}"
do
	for delta in "${delta_s[@]}" 
	do
		for ((i=1;i<seeds;i+=1))
		do
			r_min=$(expr -$r_max)
			python main.py --difficulty=maze_easy --seed=$i --domain=point --terminal --clip_state --alg=g-oac --delta=$delta --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --policy_weight_decay=3e-5 --qf_lr=3e-4 --std_lr=3e-4 --expl_policy_lr=3e-4 --r_min=$r_min --r_max=$r_max --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=h002/delta-"${delta}"_r-"${r_max}" &
		done 
	done
done
```

![image](/data/point/h002/h002-plot.png)

![image](/data/point/h002/delta-0.9_r-0.3/delta-0.9_r-0.3-plot-sep.png)

![image](/data/point/h002/delta-0.9_r-0.3/s1/heatmaps/hm_299.png)

![image](/data/point/h002/delta-0.9_r-0.3/s2/heatmaps/hm_299.png)

![image](/data/point/h002/delta-0.9_r-0.3/s3/heatmaps/hm_299.png)


# MORE


# SAC - tuning on layers size

Higher layers sizes show significantly better results, this si a problem if we want to run on turing.

Also as soon as I scaled the reward SAC behavior changed significantly, SAC paper says it is an importan hp.

```sh
lr_s=(1e-4 3e-4 10e-4)
ls_s=(32 64 128 256)

for lr in "${lr_s[@]}"
do
	for ls in "${ls_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
			taskset -ca 22-65 python main.py --seed=$i --domain=point --terminal --clip_state --difficulty=maze_simple --alg=sac --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=$ls --policy_lr=$lr --qf_lr=$lr --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=f003/lr"${lr}"_ls"${ls}" &
		done # WATCHOUT suffix
	done
done
```

![image](/data/point/f003/f003-plot.png)

![image](/data/point/f003/f003-plot-lsh.png)

![image](/data/point/f003/f003-plot-lsl.png)

One of the best one, in term of compromise between exploration and convergence to good results is the default `lr=3e-4, ls=256`

![image](/data/point/f003/lr3e-4_ls256/lr3e-4_ls256-plot-sep-lsl.png)

![image](/data/point/f003/lr3e-4_ls256/s1/heatmaps/hm_299.png)

![image](/data/point/f003/lr3e-4_ls256/s2/heatmaps/hm_299.png)

![image](/data/point/f003/lr3e-4_ls256/s3/heatmaps/hm_299.png)

# h001 - search OAC

```sh
delta_s=(14 18 23 27)
beta_s=(3 4 5 6)

for delta in "${delta_s[@]}"
do
	for beta in "${beta_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
			taskset -ca 22-65 python main.py --seed=$i --domain=point --terminal --clip_state --difficulty=maze_simple --alg=oac --delta=$delta --beta_UB=$beta --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=h001/delta"${delta}"_beta"${beta}" &
		done # WATCHOUT suffix
	done
done
```

# h002 - search goac

Wrong maze
```sh
delta_s=(0.6 0.75 0.9)
r_max_s=(0.01 0.03 0.1 0.3)

for r_max in "${r_max_s[@]}"
do
	for delta in "${delta_s[@]}" 
	do
		for ((i=1;i<seeds;i+=1))
		do
			r_min=$(expr -$r_max)
			python main.py --difficulty=maze_easy --seed=$i --domain=point --terminal --clip_state --alg=g-oac --delta=$delta --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --policy_weight_decay=3e-5 --qf_lr=3e-4 --std_lr=3e-4 --expl_policy_lr=3e-4 --r_min=$r_min --r_max=$r_max --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=h002/delta-"${delta}"_r-"${r_max}" &
		done
	done
done
```

Right maze
```sh
delta_s=(0.6 0.75 0.9)
r_max_s=(0.01 0.03 0.1 0.3)

for r_max in "${r_max_s[@]}"
do
	for delta in "${delta_s[@]}" 
	do
		for ((i=1;i<seeds;i+=1))
		do
			r_min=$(expr -$r_max)
			python main.py --difficulty=maze_simple --seed=$i --domain=point --terminal --clip_state --alg=g-oac --delta=$delta --mean_update --max_path_length=300 --num_layers=2 --layer_size=256 --batch_size=256 --expl_policy_std=0.05 --policy_activation=LeakyReLU --policy_weight_decay=3e-5 --qf_lr=3e-4 --std_lr=3e-4 --expl_policy_lr=3e-4 --r_min=$r_min --r_max=$r_max --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --snapshot_mode=none --save_heatmap --epochs=300 --no_gpu --suffix=h002ms/delta-"${delta}"_r-"${r_max}" &
		done 
	done
done
```


# SAC - tuning on layers size

Terminal bug

```sh
lr_s=(1e-4 3e-4 10e-4)
ls_s=(32 64 128 256)

for lr in "${lr_s[@]}"
do
	for ls in "${ls_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
			taskset -ca 22-65 python main.py --seed=$i --domain=point --terminal --clip_state --difficulty=maze_simple --alg=sac --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=$ls --policy_lr=$lr --qf_lr=$lr --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=f003/lr"${lr}"_ls"${ls}" &
		done # WATCHOUT suffix
	done
done
```









