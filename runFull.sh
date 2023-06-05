# run.sh # -----------------------------------

#!/bin/bash

seed_num=3
let "seed_num++"

qty_s=(0.6 1 1.5) 

for qty in "${qty_s[@]}"
do
	for ((seed=1;seed<seed_num;seed+=1))
	do
		qsub -v seed=$seed,qty=$qty run1.sub 
	done
done


# run.sub # ----------------------------------

#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,walltime=8:00:00 -q test
#PBS -N lq19sac
# you can set the stdout and stderr file here

cd '/u/archive/laureandi/sacco/oac-explore'
source /u/archive/laureandi/sacco/wac_env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/sacco/.mujoco/mujoco200/bin

python main.py --seed=0 --domain=humanoid --alg=gs-oac --delta=0.9 --mean_update --r_min=0 --r_max=10 --prv_std_qty=$qty --prv_std_weight=0.6 --epochs=5000 --suffix=hWAC_qty-"${qty}" --no_gpu > ./data/hWAC_qty-s"${seed}"_q"${qty}" 2>&1


