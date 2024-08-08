#!/bin/bash

input_folder="../../data"
input_files=("Report(Torrevilla2023)" "Report(Barbieri2023)" "Report(Bertelegni2023)" "Report(Rebutti2023)")
lengths=(3 6 12 24 48)
algorithms=("Saits" "TimesNet" "Nonstationary_Transformer" "Autoformer" "DynaMMo" "SoftImpute" "CDMissingValueRecovery" )
percs=(0 0.5 1.0)
random_seeds=(2023 2024 2025)
input_json="hypertuning_results.json"

for algorithm in "${algorithms[@]}"; do
  echo "Running Python script with algorithm: $algorithm"
  for input_file in "${input_files[@]}"; do
    echo "Running Python script with dataset: $input_file"
    for length in "${lengths[@]}"; do
      echo "Running Python script with length: $length"
      for perc in "${percs[@]}"; do
        echo "Running Python script with perc: $perc"
        for seed in "${random_seeds[@]}"; do
          echo "Running Python script with seed: $seed"
          python3 test_imputation.py $algorithm $input_folder $input_file $length $perc $seed $input_json "${input_files[@]}"
        done
      done
    done
  done
done