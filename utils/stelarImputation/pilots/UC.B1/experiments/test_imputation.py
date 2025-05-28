from stelarImputation import imputation as tsi
from utils import test_imputation_metrics
import json
import optuna
from optuna.samplers import TPESampler
import os
import sys
import numpy as np
import time

algorithm = sys.argv[1]  # 'SoftImpute'
input_file_path = sys.argv[2]  # '../../data/sample_33UVP_lai_field_timeseries.csv'
seed = int(sys.argv[3])  # '2023'
input_json = sys.argv[4]  # 'SoftImpute'

with open(input_json) as o:
    best_params = json.load(o)

dataset_name = os.path.splitext(os.path.basename(input_file_path))[0]
params = best_params[dataset_name]

results = test_imputation_metrics(input_file_path, algorithm, params, seed)

results_file = f"experiment_results/{dataset_name}/{algorithm}/seed={seed}.json"
folder_path, file_name = os.path.split(results_file)

# Create intermediate folders if they don't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if os.path.exists(results_file):
    os.remove(results_file)

# Serializing json
json_object = json.dumps(results, indent=4)

# Writing to sample.json
with open(results_file, "w") as outfile:
    outfile.write(json_object)
