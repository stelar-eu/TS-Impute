#!/bin/bash

input_folder="../../data"
input_files=("Report(Torrevilla2023)" "Report(Barbieri2023)" "Report(Bertelegni2023)" "Report(Rebutti2023)")
length=24
perc=0
algorithms=("Naive" "SoftImpute" "CDMissingValueRecovery" "Saits" "TimesNet" "Nonstationary_Transformer" "Autoformer" "DynaMMo")
input_json="input.json"

for input_file in "${input_files[@]}"; do
  echo "Running Python script with dataset: $input_file"
  for algorithm in "${algorithms[@]}"; do
    echo "Running Python script with algorithm: $algorithm"
    python3 optuna_imputation.py $input_folder $input_file $perc $length $algorithm $input_json "${input_files[@]}"
  done
done

python3 create_json.py "$(echo "${input_files[@]}" | tr ' ' ',')" "$(echo "${algorithms[@]}" | tr ' ' ',')"