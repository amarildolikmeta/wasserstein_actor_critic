# SAC - tuning on layers size

![image](../data/point/maze_simple/maze_simple.png)

```sh
lr_s=(1e-4 3e-4 10e-4)
ls_s=(32 64 128 256)
```

Ho runnato SAC sia con che senza stati terminali su la griglia di valori riportata sopra


Senza stati terminali:
![image](../data/point/f003/f003-plot.png)

Con stati terminali:
![image](../data/point/i001/i001-plot.png)

In entrambi i casi l'unica run ad arrivare infondo è con 256 neuroni e lr 1e-3 (10e-4).

Ma soltanto una delle 3 (delle 4 nel caso con stati terminali)

Senza stati terminali:
![image](../data/point/f003/lr10e-4_ls256/lr10e-4_ls256-plot-sep.png)

Con stati terminali:
![image](../data/point/i001/lr10e-4_ls256/lr10e-4_ls256-plot-sep.png)

In entrambi i casi una delle 4 run non impara quasi niente e si blocca sulla politica che va semplicemente a destra (entrambi i casi seed=2)

Senza stati terminali:
![image](../data/point/f003/lr10e-4_ls256/s2/heatmaps/hm_299.png)

Con stati terminali:
![image](../data/point/i001/lr10e-4_ls256/s2/heatmaps/hm_299.png)

Run che sono arrivati al goal (seed=3)

Senza stati terminali:
![image](../data/point/f003/lr10e-4_ls256/s3/heatmaps/hm_299.png)

Con stati terminali:
![image](../data/point/i001/lr10e-4_ls256/s3/heatmaps/hm_299.png)

-----

Una run con pochi neuroni riesce ad entrare nel buco ma fatica a rimanerci e ad avere in comportamento "stabile" una volta entrata

```sh
ls=32
lr=10e-4
```

Senza stati terminali:
![image](../data/point/f003/lr10e-4_ls32/s3/heatmaps/hm_299.png)

Con stati terminali:
![image](../data/point/i001/lr10e-4_ls32/s1/heatmaps/hm_299.png)

------

Se hai altre domande sulle run chiedimi pure

Io a questo punto opterei per reti più grandi con 128 o 256 neuroni, per il lr invece sono più indeciso, nel nostro caso l'eplorazione dovrebbe venire meno penalizzata dal lr qui secondo me entra in gioco una qualche dipendenza con l'euristica dell'entropi che fa si che l'entropia converga a valori bassi presto. Lo stesso probabilmente succede con un numreo di neuroni bassi con la quale l'entropia tende a rimanere molto alta. 

L'altra domanda è se prendere per buoni questi hyper-par di SAC o continuare con il tuning. I parametri potrebbero essere i seguenti. Sono già stati tutti testati ma con reti da 32 o 64 neuroni

  * `num_expl_steps_per_train_loop` tested =(500 1000 2000) made a big difference
  * `num_trains_per_train_loop` didn't make much of a difference
  * `batch_size` 256 which is SAC default showed the best performance on lower layer sizes
  * `policy_lr` different from `qf_lr`: no appreciable difference on lower layers size
  * `replay_buffer_size` tried 1e6 (SAC default) and 1e5 and didn't make much of difference
  * `reward_scale` should make a big difference but we could decide never to tune it 

# i002 - OAC

```sh
delta_s=()
beta_s=()

for delta in "${delta_s[@]}"
do
	for beta in "${beta_s[@]}" # WATCHOUT suffix
	do
		for ((i=1;i<seeds;i+=1))
		do
			taskset -ca 22-65 python main.py --seed=$i --domain=point --terminal --clip_state --difficulty=maze_simple --alg=oac --delta=$delta --beta_UB=$beta --max_path_length=300 --num_eval_steps_per_epoch=2000 --num_layers=2 --layer_size=256 --policy_lr=3e-4 --qf_lr=3e-4 --batch_size=256 --replay_buffer_size=1e6 --save_heatmap --epochs=300 --no_gpu --suffix=i002/delta"${delta}"_beta"${beta}" &
		done # WATCHOUT suffix
	done
done
```


