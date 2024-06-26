## Scripts for our paper ''

This repo contains scripts of simulation and real data analysis for our paper ''. 
This README file provides instructions to reproduce the result in the numerical studies section of our paper.

### Environment and Setup

We use `python==3.9`, pytorch with version `1.12.1` and numpy with version `1.21.5`. 

### Plot the fitted representation and the true representation

Setting: dimension is 2, number of sources is 8, number of targets is 1, width is 300, depth is 3, number of samples is 2000, and seed is 1. To reproduce the plot in Fig 2 (left pannel), we use the following command
```
python exp_demo_plot_repsentation.py --p 2 --r 2 --q 1 --T 8 --width 300 --depth 3 --n_source 2000 --seed 1
```

To reproduce the plot in Fig 2 (middle pannel), we use the following command
Setting: dimension is 3
```
python exp_demo_plot_repsentation.py --p 3 --r 3 --q 1 --T 8 --width 300 --depth 3 --n_source 2000 --seed 2
```

To reproduce the plot in Fig 2 (right pannel), we use the following command
Setting: dimension is 5
```
python exp_demo_plot_repsentation.py --p 5 --r 5 --q 1 --T 8 --width 300 --depth 3 --n_source 2000 --seed 3
```

### Exp 1: Performance of RTL estimator with different sample size $n$

### Exp 2: Comparison
To run a single experiment with non-linear dimension `p`, linear dimension `q`, representation dimension `r` and random seed `s`

```
python sim_main.py --p $p --q $q --r $r --seed $s --exp_id 6
```

To reproduce Figs 8 (A) (B) and 9 (A) (B), we replicated the experiment 200 times for each $p$ and save the logs

```
bash run_exp2.sh
```

Then run the plotting script

```
python plot_exp2.py
```

### Exp 4: MNIST

Run the following command to reproduce the result in Table 2 with seed 2025 and read the log file. 

```
python exp4_mnist.py
```


### Real Data Application

The following commands are used to reproduce the results in Table 1 of Supplemental Material.

To run the Pennsylvania Reemployment Bonus Experiment, set `exp_id` to 1
```
python read_data.py --exp_id 1
```

To run the House Rent Experiment, set `exp_id` to 3
```
python read_data.py --exp_id 3
```

To reproduce the Fig 10, we run the following commands
```
bash run_real_data.sh
```
