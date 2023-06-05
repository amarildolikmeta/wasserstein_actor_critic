
# Thesis Outline

* we are not striving for better sample efficiency but for being able to solve harder environments
* We solve some of the point environments that SAC and OAC cannot solve
* what other than these mazes? 
* maze with reward only at the end?
	* here it's crucial not to lose the best
* how do we expect the algorithm and its nets to act

Ideas
* add paths from the best so far policy 

----

Turing...

# Maze Simple

![image](/data/point-unN/maze_simple/terminal/maze_simple.png)

# d035 - SAC Maze simple - single run

I have done tuning but much more could be done

(I talked to Davide)

```sh
es_s=(500 1000 2000)
t_s=(500 1000 2000)
lr_s=(0.0003 0.001)
lrm_s=(1 2 3)

for ((i=1;i<seeds;i+=1))
do
	python main.py --seed=$i --domain=point --terminal --clip_state --difficulty=maze_simple --alg=sac --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=64 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e5 --save_heatmap --snapshot_mode=all --epochs=300 --no_gpu --suffix=d035 &
done
```

![image](/data/point-unN/maze_simple/terminal/sac_/d035/d035-plot-sep.png)

![image](/data/point-unN/maze_simple/terminal/sac_/d035/s1/heatmaps/hm_299.png)

![image](/data/point-unN/maze_simple/terminal/sac_/d035/s2/heatmaps/hm_299.png)

![image](/data/point-unN/maze_simple/terminal/sac_/d035/s3/heatmaps/hm_299.png)

# OAC...

# Medium

![image](/data/point-unN/medium/terminal/medium.png)

# g-oac medium - no `--mean_update`

![image](/data/point-unN/medium/terminal/g-oac_/e012/e012-plot-A.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012/e012-plot-B.png)

std should be initalized low 

Let's focus on 2.5 extremes

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.6_r-2.5/delta-0.6_r-2.5-plot-sep-r1.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.9_r-2.5/delta-0.9_r-2.5-plot-sep-r1.png)

(I forgot to enable the target policy)

## 0.9

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.9_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.9_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.9_r-2.5/s3/heatmaps/hm_299.png)

## 0.6

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.6_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.6_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012/delta-0.6_r-2.5/s3/heatmaps/hm_299.png)


# g-oac medium - `--mean_update`

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/e012mu-plot-A.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/e012mu-plot-B.png)

std should be initalized low 

Let's focus on 2.5 extremes

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.6_r-2.5/delta-0.6_r-2.5-plot-sep-r1.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.9_r-2.5/delta-0.9_r-2.5-plot-sep-r1.png)

(I forgot to enable the target policy)

## 0.9

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.9_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.9_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.9_r-2.5/s3/heatmaps/hm_299.png)

## 0.6

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.6_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.6_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e012mu/delta-0.6_r-2.5/s3/heatmaps/hm_299.png)

---<<>>----

# LOCAL

Let's focus on 2.5 extremes

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.6_r-2.5/delta-0.6_r-2.5-plot-sep-r1.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.9_r-2.5/delta-0.9_r-2.5-plot-sep-r1.png)

(I forgot to enable the target policy)

## 0.9

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.9_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.9_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.9_r-2.5/s3/heatmaps/hm_299.png)

## 0.6

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.6_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.6_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013/delta-0.6_r-2.5/s3/heatmaps/hm_299.png)


# g-oac medium - `--mean_update`

std should be initalized low 

Let's focus on 2.5 extremes

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.6_r-2.5/delta-0.6_r-2.5-plot-sep-r1.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.9_r-2.5/delta-0.9_r-2.5-plot-sep-r1.png)

(I forgot to enable the target policy)

## 0.9

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.9_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.9_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.9_r-2.5/s3/heatmaps/hm_299.png)

## 0.6

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.6_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.6_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/medium/terminal/g-oac_/e013mu/delta-0.6_r-2.5/s3/heatmaps/hm_299.png)

----<<>>---

# goac same parameters on maze easy

```sh
delta_s=(0.6 0.75 0.9)
r_max=2.5

for delta in "${delta_s[@]}" 
do
	for ((i=1;i<seeds;i+=1))
	do
		r_min=$(expr -$r_max)
		python main.py --seed=$i --domain=point --difficulty=medium --terminal --clip_state --alg=g-oac --delta=$delta --mean_update --max_path_length=300 --num_layers=2 --layer_size=64 --batch_size=128 --expl_policy_std=0.05 --policy_activation=LeakyReLU --policy_weight_decay=3e-5 --qf_lr=3e-4 --std_lr=3e-4 --expl_policy_lr=3e-4 --r_min=$r_min --r_max=$r_max --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_heatmap --epochs=300 --no_gpu --suffix=e013/delta-"${delta}"_r-"${r_max}" &
		# --mean_update
	done
done
```

# mean_update - 0.6 2.5

![image](/data/point-unN/maze_simple/terminal/mean_update_/g-oac_/e013b/delta-0.6_r-2.5/s2/heatmaps/hm_154.png)

![image](/data/point-unN/maze_simple/terminal/mean_update_/g-oac_/e013b/delta-0.6_r-2.5/s2/heatmaps/hm_155.png)

![image](/data/point-unN/maze_simple/terminal/mean_update_/g-oac_/e013b/delta-0.6_r-2.5/s2/heatmaps/hm_156.png)

![image](/data/point-unN/maze_simple/terminal/mean_update_/g-oac_/e013b/delta-0.6_r-2.5/s2/heatmaps/hm_157.png)

**The target policy is often worse than the expl policy maybe we should regualarize it too, or maybe it was just stochastic**

![image](/data/point-unN/maze_simple/terminal/mean_update_/g-oac_/e013b/delta-0.6_r-2.5/s1/heatmaps/hm_299.png)

![image](/data/point-unN/maze_simple/terminal/mean_update_/g-oac_/e013b/delta-0.6_r-2.5/s2/heatmaps/hm_299.png)

![image](/data/point-unN/maze_simple/terminal/mean_update_/g-oac_/e013b/delta-0.6_r-2.5/s3/heatmaps/hm_299.png)

If we increase the std intialization we might be able to better explore the whole but at that point we need to keep in memory that path

# 

# NEXT

A bunch of tuning could be done on parameters like

* `num_trains_per_train_loop` should change it a lot
* policy regularization
	* `policy_activation=LeakyReLU` 
	* `policy_weight_decay` 
	* `policy_grad_steps=1`
* all the parameters that were taken from SAC tuning
	* `lr`
	* `expl_policy_std`
	* `batch_size` which is 128 but SAC works better on 256 (I forgot to change it) 
	* `num_expl_steps_per_train_loop` which seems to affect the algorithm a lot
* fake updates
	* `prv_std_qty`
	* `prv_std_weight`

What should we tune? should we do it on a harder env? I think so.

Otherwise we can tune on this one trying to expect a different behaviour but that will require more run analysis maybe. 


