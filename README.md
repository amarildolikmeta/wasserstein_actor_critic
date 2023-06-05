# Wasserstein Actor Critic

This repository contains the code accompanying the NeurIPS 2022 paper 'Wasserstein Actor-Critic: Directed Exploration via Optimism for Continuous-Actions Control'.

# Running Experiments

For software dependencies, please have a look inside the requirements.txt

To run Soft Actor Critic on Humanoid with seed ```0``` as a baseline to compare against Optimistic Actor Critic, run

```
python main.py --seed=0 --domain=humanoid
```

To run Wasserstein Actor Critic on Point 4 with random seed and the hyperparameters from the paper launch:

```
python main.py --seed=0 --domain=point --terminal --clip_state --difficulty=double_L --alg=gs-oac --stable_critic --delta=0.95 --max_path_length=300 --std_lr=1e-3 --policy_lr=1e-3 --qf_lr=1e-3 --r_min=-0.5 --r_max=0.5 --prv_std_qty=0.6 --prv_std_weight=0.6 --save_heatmap --epochs=300 --no_gpu --suffix=DL3_GSWAC
```
The same file can be used to reproduce all the results, with all the environments and algorithms, by modyfying the parameter values.
