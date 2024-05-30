## Scripts for our paper ''

This repo contains scripts of simulation and real data analysis for our paper ''. 
This README file provides instructions to reproduce the result in the numerical studies section of our paper.

### Environment and Setup

We use `python==3.9`, pytorch with version `1.12.1` and numpy with version `1.21.5`. 
Other dependencies can be found in `environment.yml`. [??]

### Plot the fitted representation and the true representation

Setting: dimension is 2, number of sources is 8, number of targets is 1, width is 300, depth is 3, number of samples is 2000, and seed is 1. To reproduce the plot in Fig 2 (left pannel), we use the following command
```
demo_plot_repsentation.py --p 2 --r 2 --q 1 --T 8 --width 300 --depth 3 --n_source 2000 --seed 1
```

To reproduce the plot in Fig 2 (middle pannel), we use the following command
Setting: dimension is 3
```
demo_plot_repsentation.py --p 3 --r 3 --q 1 --T 8 --width 300 --depth 3 --n_source 2000 --seed 2
```

To reproduce the plot in Fig 2 (right pannel), we use the following command
Setting: dimension is 5
```
demo_plot_repsentation.py --p 5 --r 5 --q 1 --T 8 --width 300 --depth 3 --n_source 2000 --seed 3
```

### Exp 1: Performance of SRTR estimator with different sample size $n$

To run a single experiment with non-linear dimension `p`, linear dimension `q`, representation dimension `r` and random seed `s`

```
python sim-main.py --p $p --q $q --r $r --seed $s --exp_id 6
```

To reproduce Fig 3 (a), we first replicated the experiment 200 times for each $p$ and save the logs

```
mkdir logs/exp1-old
bash scripts/exp1.sh
```

Then plot Fig 3 (a) using the command

```
cd visualize
python exp1.py
```

To get the plot in Fig 3 (d), we use the command

```
python far_vis.py --p 1000
```


### Exp 2: Comparison with Dropout

We need to reuse the logs in Exp 1

```
cp -r logs/exp1-old logs/exp2-0
```

To replicate the experiment 

```
mkdir logs/exp2-0.3
mkdir logs/exp2-0.5
mkdir logs/exp2-0.6
mkdir logs/exp2-0.7
mkdir logs/exp2-0.8
mkdir logs/exp2-0.9
bash scripts/exp2-z3.sh
bash scripts/exp2-z5.sh
bash scripts/exp2-z6.sh
bash scripts/exp2-z7.sh
bash scripts/exp2-z8.sh
bash scripts/exp2-z9.sh
```

Visualize the result (Fig 3 (b))

```
cd visualize
python exp2.py
```


### Exp 3: When $n_1$ is large enough?

To replicate the experiment

```
mkdir logs/exp3
bash scripts/exp3.sh
```

Visualize the result (Fig 3 (c))

```
cd visualize
python exp3.py
```

### Exp 4

For the result in Fig 4 (a),

```
mkdir logs/exp4-hcm0-m200 
bash scripts/exp4-1.sh
cd visualize
python exp4-1.py
```

To reproduce the result in Fig 4 (b), we use the following command

```
mkdir logs/exp4-hcm3-m200 
bash scripts/exp4-2.sh
cd visualize
python exp4-2.py
```

To plot an visualize as Fig 5, one should use the following command

```
python fast_exp.py --hcm_id 0 --p 1000 --visualize_mat True --seed 5
python fast_exp.py --hcm_id 3 --p 1000 --visualize_mat True --seed 5
```

and the plot will be saved as `FAST-NN/a.pdf`.


### Real Data Application

The following commands are used to reproduce the results in Table 1 of Supplemental Material.

```
python fredmd_cross.py --idx $id
```

id is 88 for TB6SMFFM, 87 for TB3SMFFM, and 28 for UEMP15T26.

