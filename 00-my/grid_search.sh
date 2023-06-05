

beta_s=(3 4 5 6)
delta_s=(10 20 30 40)
# remember suffix

# triple
delta_s=(0.6 0.75 0.95)
ls_s=(64 32)
expl_policy_std_s=(0.1 0.25)

for ls in "${ls_s[@]}"
do
	for delta in "${delta_s[@]}" # WATCHOUT suffix
	do
		for expl_policy_std in "${expl_policy_std_s[@]}" # WATCHOUT suffix
		do
			for ((i=1;i<seeds;i+=1))
			do
			#taskset -ca 22-43 
				python main.py --seed=$i --domain=point --difficulty=empty --clip_state --alg=g-oac --delta=$delta --max_path_length=300 --min_num_steps_before_training=10000 --num_eval_steps_per_epoch=3000 --num_expl_steps_per_train_loop=3000 --num_layers=2 --layer_size=$ls --batch_size=32 --mean_update --expl_policy_std=$expl_policy_std --policy_activation=LeakyReLU --qf_lr=3e-4 --std_lr=1e-4 --policy_lr=1e-3 --policy_grad_steps=5 --policy_weight_decay=3e-5 --r_min=-10 --r_max=+10 --prv_std_qty=1 --dont_use_target_std --prv_std_weight=2 --replay_buffer_size=1e5  --no_gpu --snapshot_mode=gap_and_last --snapshot_gap=50 --epochs=400 --suffix=d011/delta-"${delta}"_ls-"${ls}"_expl-std-"${expl_policy_std}" --no_gpu &
			done # WATCHOUT suffix
		done
	done
done


# double
for beta in "${beta_s[@]}"
do
	for delta in "${delta_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
				taskset -ca 22-43 python main.py --domain riverswim --alg oac --num_expl_steps_per_train_loop=1000 --num_trains_per_train_loop=1000 --num_eval_steps_per_epoch 1000 --beta_UB=$beta --delta=$delta --batch_size 32 --replay_buffer_size 1e5  --epochs 200 --save_sampled_data --snapshot_gap=25 --snapshot_mode=gap_and_last --keep_first=5 --suffix 009/delta-"${delta}"_beta-"${beta}" --no_gpu &
		done # WATCHOUT suffix
	done
done


# single
lrs=(5e-5 7e-5 10e-5 15e-5 20e-5)

for lr in "${lrs[@]}" # WATCHOUT suffix
do
	for ((i=1;i<seeds;i+=1))
	do
		python main.py --seed=$i --domain=riverswim --deterministic_rs --alg=g-oac --delta=0.95 --max_path_length 30 --num_layers=1 --layer_size=16 --std_lr=$lr --r_min=0 --r_max=1 --replay_buffer_size=1e5 --save_sampled_data --snapshot_gap=5 --snapshot_mode=gap_and_last --keep_first=20 --no_gpu --epochs=250 --suffix=b001/lr-"${lr}" &
	done # WATCHOUT suffix
done

# no grid
for ((i=1;i<seeds;i+=1))
do
	python  &
done

