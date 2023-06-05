

## 001 - Amarildo suggested run on riverswim with r_min =0
```sh
for ((i=0;i<5;i+=1))
do
	taskset -ca 0-20 python main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --counts --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --no_gpu --epochs=2000 &
done
```
* after this one gridsearch on r_min
* weird behaviour on uncertainty let's first try without counts to see if that was the issue
* ![image](/data/riverswim/25/001/001_count.png)

## 002 - same no counts
```sh
for ((i=0;i<5;i+=1))
do
	python main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --save_sampled_data --snapshot_gap=10 --r_min=0 --r_max=0.1 --snapshot_mode=gap_and_last --no_gpu --epochs=2000 &
done
```
![image](/data/riverswim/25/002b/002-3sep.png)
* 2 runs didn't learn anything, might need higher r_max
* weird behviour, heatmaps of the second run

50
![image](/data/riverswim/25/002b/02-up/heatmaps_rel/hm_50.png)
80
![image](/data/riverswim/25/002b/02-up/heatmaps_rel/hm_80.png)
240
![image](/data/riverswim/25/002b/02-up/heatmaps_rel/hm_240.png)
540
![image](/data/riverswim/25/002b/02-up/heatmaps_rel/hm_540.png)
900
![image](/data/riverswim/25/002b/02-up/heatmaps_rel/hm_900.png)
940
![image](/data/riverswim/25/002b/02-up/heatmaps_rel/hm_940.png)
1000
![image](/data/riverswim/25/002b/02-up/heatmaps_rel/hm_1000.png)


## 003 - mean update

```sh
for ((i=0;i<5;i+=1))
do
	python main.py --seed=$i --domain riverswim --alg p-oac --mean_update --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --no_gpu --epochs=2000 &
done
```
![image](/data/riverswim/25/003-mu/003-plot.png)
* similar results

## 005 - p-oac (2 layers)

```sh
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers --num_layers 2 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --no_gpu --epochs=2000 &
done
```
![image](/data/riverswim/25/005-poac2L/plot-005.png)
Std
![image](/data/riverswim/25/005-poac2L/std-005.png)
Focus on black run (unintelligible heatmaps)

## 006 - p-oac (2 estimators)

```sh
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 2 --delta 0.95 --max_path_length 100 --share_layers --num_layers 2 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=2000 --log_dir=data/006 &
done
```
![image](/data/riverswim/25/006-poac2e/006.png)

## 007 - p-oac no bias training

```sh
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 2 --delta 0.95 --max_path_length 100 --share_layers --num_layers 2 --layer_size 16 --no_train_bias --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=2000 --log_dir=data/006 &
done
```
* [ ] converges really fast
* [ ] but it does not converge to the optimal meanQ how? 

![image](/data/riverswim/25/007-notb/007notb.png)

`
./data/006/riverswim/25/p-oac_/1637278392.4458997
first optimal iteration: itr_20.zip_pkl
~expl steps before convergence: 40000
`

## g004 - g-oac

```sh
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=2000 --log_dir=data/004b &
done
```
* running (almost)
![image](/data/riverswim/25/g004-g-oac_/g004-plot.png)

## g008 - weak g-oac

```sh
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg g-oac --num_expl_steps_per_train_loop=500 --num_trains_per_train_loop=300 --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=1000 --log_dir=data/g008 &
done
```
* running (almost)
![image](/data/riverswim/25/g008-goac-weak/g008-plot.png)

## 009-OAC

grid search wit OAC
```sh
betas=(3 4 5 6)
deltas=(10 20 30 40)
# remember suffix
for beta in "${betas[@]}"
do
  for delta in "${deltas[@]}" # WATCHOUT suffix
  do
      for ((i=0;i<3;i+=1))
      do
          taskset -ca 22-43 python main.py --domain riverswim --alg oac --num_expl_steps_per_train_loop=1000 --num_trains_per_train_loop=1000 --num_eval_steps_per_epoch 1000 --beta_UB=$beta --delta=$delta --batch_size 32 --replay_buffer_size 1e5  --epochs 200 --save_sampled_data --snapshot_gap=25 --snapshot_mode=gap_and_last --keep_first=5 --suffix 009/delta_"${delta}"_beta_"${beta}"/ --no_gpu &
      done # WATCHOUT suffix
  done
done
```



