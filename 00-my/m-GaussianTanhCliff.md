
# solve policy fit

# run SAC 

```sh
for ((seed=1;seed<seed_num;seed+=1))
do
  python main.py --seed=$seed --domain=cliff_mono --alg=sac --max_path_length=50 --save_sampled_data --layer_size=128 --save_heatmap --save_sampled_data --no_gpu --epochs=25 --suffix=a01 &
done
```

solves it

![image](/data/cliff_mono/sac_/a01/s1/heatmaps/hm_24.png)

![image](/data/cliff_mono/sac_/a01/a01-plot-sep-grads.png)

# run SAC random

```sh
for ((seed=1;seed<seed_num;seed+=1))
do
  python main.py --seed=$seed --domain=cliff_mono --alg=sac --random_policy --max_path_length=50 --save_sampled_data --layer_size=128 --save_heatmap --save_sampled_data --no_gpu --epochs=25 --suffix=a01r &
done
```

![image](/data/cliff_mono/sac_/a01r/s1/heatmaps/hm_24.png)

![image](/data/cliff_mono/sac_/a01r/a01r-plot-sep-grads.png)

solves it

# G-OAC `--policy_output=GaussianTanh`

```sh
for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=$seed --domain=cliff_mono --alg=g-oac --random_policy --delta=0.9 --max_path_length=50 --layer_size=128 --expl_policy_std=0.05 --mean_update --std_lr=3e-4 --policy_output=GaussianTanh --r_min=-1 --r_max=0 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --save_sampled_data --save_heatmap --epochs=25 --no_gpu --suffix=a02gt &
done
```

# G-OAC `--policy_output=TanhGaussian`

```sh
for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=$seed --domain=cliff_mono --alg=g-oac --random_policy --delta=0.9 --max_path_length=50 --layer_size=128 --expl_policy_std=0.05 --mean_update --std_lr=3e-4 --r_min=-1 --r_max=0 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --save_sampled_data --save_heatmap --epochs=25 --no_gpu --suffix=a02tg &
done
```











# G-OAC `--policy_output=GaussianTanh`

```sh
for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=$seed --domain=cliff_mono --alg=g-oac --delta=0.9 --max_path_length=50 --layer_size=128 --expl_policy_std=0.05 --mean_update --std_lr=3e-4 --policy_output=GaussianTanh --r_min=-1 --r_max=0 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --save_sampled_data --save_heatmap --epochs=25 --no_gpu --suffix=a02 &
done
```

bad gradients, doesn't solve

# patch

```patch
diff --git a/trainer/gaussian_trainer.py b/trainer/gaussian_trainer.py
index b5c432d..1253307 100644
--- a/trainer/gaussian_trainer.py
+++ b/trainer/gaussian_trainer.py
@@ -70,7 +70,8 @@ class GaussianTrainer(SACTrainer):
                          target_update_period=target_update_period,
                          use_automatic_entropy_tuning=use_automatic_entropy_tuning,
                          target_entropy=target_entropy,
-                         deterministic=deterministic)
+                         #deterministic=deterministic)
+                         deterministic=False)
 
         self.action_space = action_space
         self.q_min = q_min
```

#  G-OAC `--policy_output=GaussianTanh` non det policy

```sh
for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=$seed --domain=cliff_mono --alg=g-oac --delta=0.9 --max_path_length=50 --layer_size=128 --expl_policy_std=0.05 --mean_update --std_lr=3e-4 --policy_output=GaussianTanh --r_min=-1 --r_max=0 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --save_sampled_data --save_heatmap --epochs=25 --no_gpu --suffix=a03 &
done
```

#  G-OAC `--policy_output=GaussianTanh` non det policy

```sh
for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=$seed --domain=cliff_mono --alg=g-oac --delta=0.9 --random_policy --max_path_length=50 --layer_size=128 --expl_policy_std=0.05 --mean_update --std_lr=3e-4 --policy_output=GaussianTanh --r_min=-1 --r_max=0 --prv_std_qty=0.3 --dont_use_target_std --prv_std_weight=1 --save_sampled_data --save_heatmap --epochs=25 --no_gpu --suffix=a04 &
done
```

