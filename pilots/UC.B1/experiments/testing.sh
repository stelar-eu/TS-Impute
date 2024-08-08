#!/bin/bash

input_files=("../../data/UCB1_lai_field_timeseries.csv")
algorithms=("Naive" "SoftImpute" "CDMissingValueRecovery" "Saits" "TimesNet")
random_seeds=(2023 2024 2025)
input_json="hypertuning_results.json"

for algorithm in "${algorithms[@]}"; do
  echo "Running Python script with algorithm: $algorithm"
  for input_file in "${input_files[@]}"; do
    echo "Running Python script with dataset: $input_file"
    for seed in "${random_seeds[@]}"; do
      echo "Running Python script with seed: $seed"
      python3 test_imputation.py $algorithm $input_file $seed $input_json
    done
  done
done