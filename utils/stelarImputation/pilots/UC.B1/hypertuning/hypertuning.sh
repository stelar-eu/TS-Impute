#!/bin/bash

input_folder="../../data"
input_files=("UCB1_lai_field_timeseries")
algorithms=("Naive" "SoftImpute" "CDMissingValueRecovery" "Saits" "TimesNet")
input_json="input.json"

for input_file in "${input_files[@]}"; do
  echo "Running Python script with dataset: $input_file"
  for algorithm in "${algorithms[@]}"; do
    echo "Running Python script with algorithm: $algorithm"
    python3 optuna_imputation.py $input_folder $input_file $algorithm $input_json
  done
done

python3 create_json.py "$(echo "${input_files[@]}" | tr ' ' ',')" "$(echo "${algorithms[@]}" | tr ' ' ',')"