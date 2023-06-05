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

## 32

with fewer neurons the algorithm tends to explore much more, to the point where it does enter inside the hole but it can't adapat the q function.

It actually has problem in fitting the much easier local optimum that the other ones find.

![image](/data/point/f003/lr3e-4_ls32/s1/heatmaps/hm_299.png)

![image](/data/point/f003/lr3e-4_ls32/s2/heatmaps/hm_299.png)

![image](/data/point/f003/lr3e-4_ls32/s3/heatmaps/hm_299.png)

# Weird change on overleaf default

* `batch_size` = 32
* `replay_buffer_size` = 1e4

# ----------------------------------

* find the environments or think about the environments we want the algorithm to solve

* find an environment harder than riverswim similar to those environments but that we can observe and tune our algorithm on it.
	* do we still have regularization and policy fitting problems?
	* do we need to regualarize the target policy

* But first we need to define how we want it to work.
	* I still lean toward having a stable `target_policy`/`Mean network` 

* define the KPI's, ok the return but we need to see some behaviours as well

* a schedule and method on how to deal with hyperparamters
	* try with only 2 different values and see what hp make the biggest difference
	* try to tune those
	* try to tune the other ones with those ones fixed

* if I use something more elaborate thant mere grid search I could add it to the thesis

* then we can start to not very clean adjustments to make it more emperically effective

# GOAC

```sh
for ((i=1;i<seeds;i+=1))
do
	python main.py --seed=$i --domain=point --difficulty=medium --terminal --clip_state --alg=g-oac --delta=0.75 --mean_update --max_path_length=300 --num_layers=2 --layer_size=64 --batch_size=128 --expl_policy_std=0.05 --policy_activation=LeakyReLU --policy_weight_decay=3e-5 --qf_lr=3e-4 --std_lr=3e-4 --expl_policy_lr=3e-4 --r_min=-0.2 --r_max=0.2 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_heatmap --epochs=200 --no_gpu --suffix=f002 &
	# --mean_update
done
```





