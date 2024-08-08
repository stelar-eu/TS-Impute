# Hyper-Parameter Tuning

The performance of the algorithms and especially the ML-Based ones 
highly depend on the settings of hyper-parameters. 
To make fair comparisons across imputation methods and draw impartial 
conclusions, we run each algorithm 50 times for the dataset 
(i.e., tune it with 50 groups of hyper-parameters) and use the 
best configuration to conduct the final evaluation and get the 
reported results. The Linear interpolation method was not 
included in this hyper-parameter tuning.

## Instructions
Install optuna library:
```sh
$ pip install optuna
```
 
Make the necessary changes in hypertuning.sh conserving the folder
path and the datasets (must be .CSV files in the format provided by 
the time series extraction module).

**_NOTE:_** We have datasets with many fields and a percentage 
of missing values higher than 27%. 
So, for each dataset, we create a new smaller dataset whose name 
starts with 'sample_' and contains 10% of the original number of 
fields that have the smallest number of missing values.

Then, execute hyper-parameter tuning with the following commands:
```sh
$ chmod +x hypertuning.sh
$ ./hypertuning.sh
```

We can find the JSON with the best combination of hyper-parameters 
for each algorithm in the 'experiments' folder outside the current
folder.

