#!/bin/bash
# chmod +x ex_server.sh

# 1st column
## taskset -ca 0-21
# 2nd column
## taskset -ca 22-43
# 3rd column
## taskset -ca 44-65
# 4th column
## taskset -ca 66-88

# --seed=$1 x
# no &
# no export

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/particle/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/particle/.mujoco/mujoco200/bin

# seed_num=1
# let "seed_num++"
#seed_s=(672272)
#seed_s=(62386 333097 491959 494518 546394 672272 801228 977172)

# seed_s=(333097 546394 672272 801d28 977172)

seed_num=3
let "seed_num++"

for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=0 --domain=walker2d --alg=gs-oac --stable_critic --delta=0.9 --mean_update --r_min=-0.5 --r_max=0.5 --prv_std_qty=1 --prv_std_weight=0.6 --epochs=5000 --no_gpu &
  python main.py --seed=0 --domain=walker2d --alg=sac  --epochs=5000 --no_gpu &
  python main.py --seed=0 --domain=walker2d -beta_UB=4.66 --delta=23.53  --epochs=5000 --no_gpu &
done


for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=0 --domain=hopper --alg=gs-oac --stable_critic --delta=0.9 --mean_update --r_min=-0.5 --r_max=0.5 --prv_std_qty=1 --prv_std_weight=0.6 --epochs=2000 --no_gpu &
  python main.py --seed=0 --domain=hopper --alg=sac  --epochs=2000 --no_gpu &
  python main.py --seed=0 --domain=hopper -beta_UB=4.66 --delta=23.53  --epochs=2000 --no_gpu &
done

for ((seed=1;seed<seed_num;seed+=1))
do
	python main.py --seed=0 --domain=halfcheetah --alg=gs-oac --stable_critic --delta=0.9 --mean_update --r_min=-0.5 --r_max=0.5 --prv_std_qty=1 --prv_std_weight=0.6 --epochs=5000 --no_gpu &
  python main.py --seed=0 --domain=halfcheetah --alg=sac  --epochs=5000 --no_gpu &
  python main.py --seed=0 --domain=halfcheetah -beta_UB=4.66 --delta=23.53  --epochs=5000 --no_gpu &
done



