# Experiments

For each imputation method we used the best combination of 
hyper-parameters for each dataset, as derived after the 
hyper-parameter tuning. For each combination of dataset 
and imputation method, we perform three iterations using 
three different random seeds. Each random seed picks a different
time-point across the series where its LAI value is dropped 
and considered as missing for the evaluation. 
The experiments for a given dataset and random seed generate missing 
values in the same time-points; this dataset is then given as
input to the examined imputation methods.

## Instructions

First, make the necessary changes in testing.sh conserving the folder
path and the datasets (must be .CSV files in the format provided by 
the time series extraction module).

Execute the experiments with the following commands:
```sh
$ chmod +x testing.sh
$ ./testing.sh
```

The experiment_results folder contains the metrics for each of the experiments
on each of the algorithms for each of the datasets.