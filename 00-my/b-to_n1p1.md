
# difference between states [0,25] and [-1,+1]

```sh
python main.py --seed=0 --domain riverswim --alg g-oac --delta=0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --r_min=0 --r_max=0.1 --save_sampled_data --snapshot_mode=all --no_gpu --epochs=20 

--suffix=ff00-1 # either 
# change code
--suffix=ff00+1 # or
```

## ff00 - [0,25]

### 0-25

always: -1

![image](/data/riverswim/25/fakes-goac/ff00-1/1637789143.007575/heatmaps/hm_19.png)

always: +1

![image](/data/riverswim/25/fakes-goac/ff00+1/1637788334.166294/heatmaps/hm_19.png)

### -1 : +1

always: -1

![image](/data/riverswim/25/fakes-goac/ff00-1/1637789143.007575/heatmaps-1+1/hm_19.png)

always: +1

![image](/data/riverswim/25/fakes-goac/ff00+1/1637788334.166294/heatmaps-1+1/hm_19.png)

## ff01 [-1,+1]

### 0-25

always: -1

![image](/data/riverswim/25/fakes-goac/ff01-1/1637797383.738532/heatmaps-25/hm_19.png)

always: +1

![image](/data/riverswim/25/fakes-goac/ff01+1/1637797006.478863/heatmaps-25/hm_19.png)

### -1 : +1

always: -1

![image](/data/riverswim/25/fakes-goac/ff01-1/1637797383.738532/heatmaps-1+1/hm_19.png)

always: +1

![image](/data/riverswim/25/fakes-goac/ff01+1/1637797006.478863/heatmaps-1+1/hm_17.png)


