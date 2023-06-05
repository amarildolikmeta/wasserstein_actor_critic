
<!-- 
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_0.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_1.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_2.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_3.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_4.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_5.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_6.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_7.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_8.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_9.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_10.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_11.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_12.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_13.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_14.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_15.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_16.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_17.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_18.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_19.png)
![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_20.png)
-->

## nc01 `--r_min=-0.1 --r_max=0`

```sh
python3 main.py --seed=$1 --domain cliff_mono --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=-0.1 --r_max=0 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=200 --suffix=nc01
```


![image](/data/cliff_mono/g-oac_/nc01/nc01-plot-sep.png)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_0.png)

having fallen in the cliff the mean shifts to the wrong size and it now also prefers to get away from the cliff

std also shifts to the wrong side
![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_1.png)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_2.pn)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_3.pn)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_4.pn)


std starts shifting
![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_5.png)

the mean shifts down and the critic now suggests to move right
![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_6.png)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_7.png)


the agent manages to get to the end and the mean shifts to the right
![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_8.png)

the bottom policy inexplicably shifts up (maybe the UB is moving up and down)
![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_9.png)

the critic prefers positive non +1 action on the right states, this might due to the fact that those are the only ones it's got
![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_10.png)


having falling into the cliff multiple times the critic now goes back to -1 action, although he still kinds of want to move right once it surpasses the cliff
![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_11.png)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_12.pn)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_13.pn)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_14.pn)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_15.pn)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_16.png)

Inexplicably the mean shifts down

![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_17.png)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_18.png)

the agents falls into the cliff and all comes back up

![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_19.png)


![image](/data/cliff_mono/g-oac_/nc01/s242318/heatmaps_det/hm_20.png)


## nc01-0.4+0 `--r_min=-0.4 --r_max=0`
```sh
python3 main.py --seed=$1 --domain cliff_mono --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=-0.4 --r_max=0 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=200 --suffix=nc01-0.4+0
```
![image](/data/cliff_mono/g-oac_/nc01-.4+0/nc01-.4+0-plot-sep.png)

## c006ntptl: more training steps (`num_trains_per_train_loop`)

```sh
python3 main.py --seed=0 --domain cliff_mono --alg g-oac --dim=12 --delta=0.95 --max_path_length 100 --num_trains_per_train_loop=2000 --share_layers --num_layers 1 --layer_size 16 --r_min=-0.1 --r_max=0 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=20 --suffix=c006ntptl
```

![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_0.png)
Thanks to the policy learned at 0 the agent gets to the end many times


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_1.png)
The policy has changed and now the agent falls into the cliff


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_2.png)


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_3.png)
having fallen into the cliff several times the egents learns a maximum in 0,-1


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_4.png)
the critic tells the policy to get away from the cliff


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_5.png)


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_6.png)


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_7.png)


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_8.png)


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_9.png)
it takes the std several iterations to be less than other parts in 0,-1


![image](/data/cliff_mono/g-oac_/c006ntptl/s105685/heatmaps_det/hm_10.png)



## c006mu `--mean_updates`

```sh
python3 main.py --seed=0 --domain cliff_mono --alg g-oac --dim=12 --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=-0.1 --r_max=0 --mean_update --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=20 --suffix=c006mu
```

## c007go `--global_opt` 

```sh
python3 main.py --seed=0 --domain cliff_mono --alg g-oac --dim=12 --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=-0.1 --r_max=0 --global_opt --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=20 --suffix=c007go-sf
```

![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_0.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_0.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_1.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_1.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_2.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_2.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_3.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_3.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_4.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_4.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_5.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_5.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_6.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_6.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_7.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_7.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_8.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_8.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_9.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_9.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_10.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_10.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_11.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_11.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_12.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_12.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_13.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_13.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_14.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_14.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_15.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_15.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_16.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_16.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_17.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_17.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_18.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_18.jpg)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/heatmaps/hm_19.png)
![image](/data/cliff_mono/g-oac_/c007go-sf/s323579/upper_bound_policy_opt_19.jpg)

## c007go40 `--global_opt` 40 backward step iter 

```sh
python3 main.py --seed=0 --domain cliff_mono --alg g-oac --dim=12 --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=-0.1 --r_max=0 --global_opt --save_fig --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=20 --suffix=c007go40
```


but now the save figs don't make as much sense as they plot the last iteratio![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_0.png)

![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_0.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_1.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_1.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_2.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_2.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_3.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_3.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_4.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_4.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_5.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_5.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_6.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_6.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_7.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_7.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_8.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_8.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_9.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_9.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_10.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_10.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_11.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_11.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_12.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_12.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_13.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_13.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_14.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_14.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_15.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_15.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_16.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_16.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_17.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_17.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_18.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_18.jpg)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/heatmaps/hm_19.png)
![image](/data/cliff_mono/g-oac_/c007go40/s757392/upper_bound_policy_opt_19.jp


## c007sgd SGD opt
no branch modified on gaussian trainer

```sh
python3 main.py --seed=0 --domain cliff_mono --alg g-oac --dim=12 --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=-0.1 --r_max=0 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=20 --suffix=c007sgd
```

## C009 - 2 layers

```sh
python3 main.py --seed=1 --domain cliff_mono --alg g-oac --dim=12 --delta=0.95 --max_path_length 100 --share_layers --num_layers 2 --layer_size 16 --r_min=-0.1 --r_max=0 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=20 --suffix=c009-2l
```
