
`--snapshot_gap=5 --snapshot_mode=gap_and_last`

## b017

```sh
python main.py --seed=1 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=1e-4 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --keep_first=100 --no_gpu --epochs=100 --suffix=b017
```
bad fitting between 3 and 25

## b018 - global opt

```sh
python main.py --seed=1 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --global_opt --num_layers=1 --layer_size=16 --std_lr=1e-4 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b018
```
extremely bad fitting

## b019 - policy lr

```sh
lrs=(1e-4 1e-3 5e-3 1e-2)

for lr in "${lrs[@]}" # WATCHOUT suffix
do
	python main.py --seed=1 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=1e-4 --policy_lr=$lr --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b019/p-lr_"${lr}" &
done
```


## b020 - more (20) steps

```sh
python main.py --seed=1 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=1e-4 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b020
```

# b21 - search policy lr and number of steps

```sh
lrs=(1e-4 3e-4 1e-3 3e-3 1e-2)
steps=(1 5 10 20 30)

for lr in "${lrs[@]}" # WATCHOUT suffix
do
  for step in "${steps[@]}" # WATCHOUT suffix
  do
      for ((i=0;i<3;i+=1))
      do
		python main.py --seed=$i --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_lr=$lr --policy_grad_steps=$step --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b021/p-lr_"${lr}"_s_"${step}" &
      done # WATCHOUT suffix
  done
done
```
## p-lr_1e-2_s_30 / p-lr_1e-2_s_20 / p-lr_1e-2_s_10

flat on top or bottom **discard**

## p-lr_1e-2_s_5

### s1
from 20 on it doesn't move, otherwise good
![image](/data/riverswim/25/g-oac_/b021/p-lr_1e-2_s_5/s1/heatmaps/hm_20.png)
### s2
from 9 it learns a steps it won't give up
![image](/data/riverswim/25/g-oac_/b021/p-lr_1e-2_s_5/s2/heatmaps/hm_14.png)
### s0 
Good except
![image](/data/riverswim/25/g-oac_/b021/p-lr_1e-2_s_5/s575698/heatmaps/hm_16.png)
![image](/data/riverswim/25/g-oac_/b021/p-lr_1e-2_s_5/s575698/heatmaps/hm_17.png)
and 
![image](/data/riverswim/25/g-oac_/b021/p-lr_1e-2_s_5/s575698/heatmaps/hm_19.png)
![image](/data/riverswim/25/g-oac_/b021/p-lr_1e-2_s_5/s575698/heatmaps/hm_20.png)

## p-lr_1e-2_s_1

Eventually good but very often late

-------

## p-lr_3e-3_s_30

### s1 
not bad, it follows using a step but otherwise good

### s2
stuck top

### s0

stuck bottom

## p-lr_3e-3_s_20

### s1

straight line moving

### s2

stuck bottom from 17

### s0

stuck line from 17

## p-lr_3e-3_s_10

### s1 

good but step stuck for a few iterations
![image](/data/riverswim/25/g-oac_/b021/p-lr_3e-3_s_10/s1/heatmaps/hm_16.png)

### s2

learns a steps that doesn't want to give up

![image](/data/riverswim/25/g-oac_/b021/p-lr_3e-3_s_10/s2/heatmaps/hm_12.png)

### s0 

learns a steps that doesn't want to give up

## p-lr_3e-3_s_5

### s1
pretty good

### s2/s0
stuck straight line / stuck step

## p-lr_3e-3_s_1

### s1
stuck step
![image](/data/riverswim/25/g-oac_/b021/p-lr_3e-3_s_1/s1/heatmaps/hm_24.png)

### s2
stuck step

### s3
stuck step

## p-lr_1e-3_s_30

good fittin sometimes (s2) stuck

![image](/data/riverswim/25/g-oac_/b021/p-lr_1e-3_s_30/s102210/heatmaps/hm_24.png)

## p-lr_1e-3_s_20

sometimes good fitting s0 stuck to line

## p-lr_1e-3_s_10


BEST: p-lr_3e-4_s_10

* [ ] global_opt
* [ ] number of steps
* [ ] policy lr

## b22
```sh
python main.py --seed=1 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b022
```

## b22b
```sh
python main.py --seed=1 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b022b
```
```sh
python main.py --seed=2 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b022b
```
```sh
python main.py --seed=604176 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b022b
```

## b23 - leaky relu - delete
```sh
python main.py --seed=2 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b023
```

## b024 - fake policy - ReLU delete
```sh
python main.py --seed=1 --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b024
```

## b025 - fake policy - leaky ReLU delete
```sh
python main.py --seed=1 --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b025
```

# b026
useless as the policy doesn't actually fit
```sh
lrs=(1e-4 3e-4 1e-3 3e-3 1e-2)
steps=(1 5 10 20 30)

for lr in "${lrs[@]}" # WATCHOUT suffix
do
  for step in "${steps[@]}" # WATCHOUT suffix
  do
		python main.py --seed=$i --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_lr=$lr --policy_grad_steps=$step --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b026/p-lr_"${lr}"_s_"${step}" &
  done
done
```

## b027 - fake policy
```sh
python main.py --seed=1 --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b027
```

## b027 - fake policy
```sh
python main.py --seed=1 --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b027
```

# b028
```sh
lrs=(1e-4 3e-4 1e-3 3e-3 1e-2)
steps=(1 5 10 20 30)

for lr in "${lrs[@]}" # WATCHOUT suffix
do
  for step in "${steps[@]}" # WATCHOUT suffix
  do
		python main.py --seed=1 --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_lr=$lr --policy_grad_steps=$step --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b028/p-lr_"${lr}"_s_"${step}" &
  done
done
```

part of the problem might be in the fact that some states are much more visited 

# b029 - `--max_path_length=30`

```sh
for ((i=0;i<3;i+=1))
do
  python main.py --seed=$i --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=30 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_grad_steps=10 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b029
done
```


# TITLE

A grid search was done on `--policy_lr=1e-3` and `--policy_grad_steps=5`
these seem to be the best parameters


# b30 b31

fake policy, the black policy is choose from i different run and it is constant. The gray one tries to fit it.

# b30

```sh
for ((i=0;i<3;i+=1))
do
  python main.py --seed=$i --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=30 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b030 &
done
```

## s1

![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_1.png)

the left part doesn't move could be because: 
* small gradient
* std changed the very last moment
* in state 0 we have many sample and we might have made the NN saturate to get ot that point

![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_2.png)

![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_3.png)

The std must have changed abruptly at the end of training because the policy resembles the UB at the previous epoch

![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_4.png)

here it doesn't fit left which is weird given that it has many sample on that side. Probably the difference is very small, or the UB has changed at the end of the iteration

![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_5.png)

in fact here it goes back up

![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_6.png)

Then it has trouble fittin this step

![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_15.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_16.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_17.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_18.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_19.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_20.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_21.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_22.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_23.png)
![image](/data/riverswim/25/g-oac_/b030/s1/heatmaps/hm_24.png)

## s2 

Seems to be unable to fit this shape

![image](/data/riverswim/25/g-oac_/b030/s2/heatmaps/hm_14.png)
![image](/data/riverswim/25/g-oac_/b030/s2/heatmaps/hm_15.png)
![image](/data/riverswim/25/g-oac_/b030/s2/heatmaps/hm_16.png)
![image](/data/riverswim/25/g-oac_/b030/s2/heatmaps/hm_17.png)
![image](/data/riverswim/25/g-oac_/b030/s2/heatmaps/hm_18.png)
![image](/data/riverswim/25/g-oac_/b030/s2/heatmaps/hm_19.png)
![image](/data/riverswim/25/g-oac_/b030/s2/heatmaps/hm_20.png)

**sometimes the policy fits well where we have lots of samples and bad everywhere else**

# b31 `--max_path_length=30`

```sh
for ((i=0;i<3;i+=1))
do
  python main.py --seed=$i --domain=riverswim --fake_policy --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=25 --suffix=b031 &
done
```

## s2 

when the std interacts with the mean a lot it make the UB more unstable and it is harder to follow it

![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_3.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_4.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_5.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_6.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_7.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_8.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_9.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_10.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_11.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_12.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_13.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_14.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_15.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_16.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_17.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_18.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_19.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_20.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_21.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_22.png)
![image](/data/riverswim/25/g-oac_/b031/s2/heatmaps/hm_23.png)

## s84639

Here the shapes are really hard to fit, can't blame the policy NN on this one 😂

![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_0.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_1.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_2.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_3.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_4.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_5.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_6.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_7.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_8.png)
![image](/data/riverswim/25/g-oac_/b031/s84639/heatmaps/hm_9.png)


# b32 -- 250 epochs non fake policy

```sh
for ((i=1;i<4;i+=1))
do
  python main.py --seed=$i --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --std_lr=3e-5 --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=250 --suffix=b032s &
done
```


The run results are pretty good but we would the variance in the states that are visited to got down faster. 

![image](/data/riverswim/25/g-oac_/b032/s1/heatmaps/hm_45.png)

Here the std is reasonable but the minimum 13 after 132 is too much

![image](/data/riverswim/25/g-oac_/b032/s1/heatmaps/hm_132.png)

also the policy learns this step function that it never wants to abandon

![image](/data/riverswim/25/g-oac_/b032/s1/heatmaps/hm_171.png)


# b33 -- 250 epochs non fake policy, `mean_update`

Let's try with `--mean_update` since the Q network is going all over the place

```sh
for ((i=1;i<4;i+=1))
do
  python main.py --seed=$i --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --mean_update --std_lr=3e-5 --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=250 --suffix=b033 &
done
```

## s1

often doesn't fit the left side

![image](/data/riverswim/25/g-oac_/b033/s1/heatmaps/hm_83.png)

## s2/s3 straight policy always

2 of the runs learn a "straight policy" and then only move it up and down

![image](/data/riverswim/25/g-oac_/b033/s2/heatmaps/hm_3.png)

So I explored the weights of the policy

And found out that after a few epochs all the first layer weights are negative and so are the first layer biases. Everything is on the left part of the ReLU activation and the NN can only change the bias, hence the "straighe policy"

```
before epoch: 4
first-weights:tensor([[-0.9594],[-1.7599],[-1.0141],[-1.1525],[-1.1407],[-1.1197],[-1.0932],[-1.1671],[-1.1212],[-1.2216],[-1.1492],[-0.9113],[-0.9680],[-1.1082],[-0.9214],[-1.0174]])
first-bias:tensor([-0.7736, -0.2755, -0.7990, -0.9289,  0.1260, -0.9026, -0.8810, -0.9389,
        -0.9038, -0.9846, -0.0255, -0.6335, -0.7769, -0.2276, -0.6387, -0.7992])
last-weights:tensor([[ 1.7969, -1.1164,  1.8196,  1.9615,  1.2632,  1.9675,  1.9354,  1.9559,
          1.9508,  1.9870,  1.1379,  1.3441,  1.7791,  1.1809,  1.4493,  1.7838]])
last-bias:tensor([0.1434])
```

# b34 -- 250 epochs non fake policy, `mean_update`, Leaky ReLU

So I tried with LeakyReLU and excude 10 runs

```sh
for ((i=1;i<10;i+=1))
do
  python main.py --seed=$i --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --mean_update --std_lr=3e-5 --policy_activation=LeakyReLU --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=20 --suffix=b034 &
done
```

The straight policy never happens in 10 runs in the first 20 epochs.

However now the problem is this step function hard to unlearn

![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_0.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_1.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_2.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_3.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_4.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_5.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_6.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_7.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_8.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_9.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_10.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_11.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_12.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_13.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_14.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_15.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_16.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_17.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_18.png)
![image](/riverswim/25/g-oac_/b034/s2/heatmaps/hm_19.png)

However now the problem is this step function hard to unlearn

I've looked at the weights and the are expectedly high when we have a step function. We could use weight decay? 

# 35 - more epochs

```sh
for ((i=1;i<5;i+=1))
do
  python main.py --seed=$i --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --mean_update --std_lr=3e-5 --policy_activation=LeakyReLU --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=250 --suffix=b035 &
done
```

Result are pretty good however s1 (26on) and s2 (115on) get stuck at a straighe policy at the bottom and top respectively

# 36 - s1 - 50 epochs (26on)

```sh
python main.py --seed=1 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --mean_update --std_lr=3e-5 --policy_activation=LeakyReLU --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=50 --suffix=b036 &
```

# 37 - s2 - 150 epochs (115on)

```sh
python main.py --seed=2 --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length=50 --num_layers=1 --layer_size=16 --mean_update --std_lr=3e-5 --policy_activation=LeakyReLU --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=150 --suffix=b037 &
```

The porblem seems once again related to big weights

<!-- after 03 -->

# 38 - non deterministic

```sh
for ((i=1;i<6;i+=1))
do
  python main.py --seed=$i --domain=riverswim --alg=g-oac --delta=0.95 --max_path_length=100 --num_layers=1 --layer_size=16 --mean_update --std_lr=3e-5 --policy_activation=LeakyReLU --policy_lr=1e-3 --policy_grad_steps=5 --r_min=0 --r_max=1 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_gap=5 --snapshot_mode=gap_and_last --keep_first=25 --no_gpu --epochs=250 --suffix=b038 &
done  
```

It all works pretty well, but again we have
* **policy stuck problem** mostly this one
* hard to unlearn steps

![image](/data/riverswim/25/mean_update_/g-oac_/b038/s3/heatmaps/hm_170.png)

