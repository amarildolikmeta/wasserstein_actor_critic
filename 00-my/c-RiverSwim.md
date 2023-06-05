
# gn001 - normalized g-oac

```sh
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=2000 --suffix=gn001 &
done
```
![image](/data/riverswim/25/gn001/gn001-plot.png)
![image](/data/riverswim/25/gn001/gn001-sep.png)
![image](/data/riverswim/25/gn001/std250.png)

# 04 - Black (Best)

iter 9

![image](/data/riverswim/25/gn001/04/heatmaps_det/hm_9.png)

iter 20

![image](/data/riverswim/25/gn001/04/heatmaps_det/hm_20.png)

## 01 - Purple (Worse)

iter 9

![image](/data/riverswim/25/gn001/01/heatmaps_det/hm_9.png)

iter 100

![image](/data/riverswim/25/gn001/01/heatmaps_det/hm_100.png)

Even when visiting the bottom a lot the std won't change

## Notes

The other runs are more like the black one

# *Regularized* - gn002ntb - gn002c - gn002rm

## no train bias
```sh
# no_train_bias
for ((i=0;i<5;i+=1))
do
	python main.py --seed=$i --domain riverswim --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --no_train_bias --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=700 --suffix=gn002ntb
done
```

## Counts
```sh
# counts: gn002c
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --counts --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=1000 --suffix=gn002c &
done
```

## r_min/max
```sh
# r_max 1
for ((i=0;i<5;i+=1))
do
	taskset -ca 22-43 python main.py --seed=$i --domain riverswim --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=1000 --suffix=gn002rm &
done
```

# no noise g-rs-003nn

```sh
for ((i=0;i<3;i+=1))
do
	python main.py --seed=$i --domain riverswim --deterministic_rs --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_gap=10 --snapshot_mode=gap_and_last --keep_first=10 --no_gpu --epochs=250 --suffix=g-rs-003nn &
done
```

