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
