from stelarImputation import imputation as tsi
from utils import imputation_metrics
import json
import optuna
from optuna.samplers import TPESampler
import os
import sys

input_folder = sys.argv[1]  # '../../data'
input_file = sys.argv[2]  # '33UVP_lai_field_timeseries'
algorithm = sys.argv[3]  # 'SoftImpute'
input_json = sys.argv[4]  # 'input.json'

with open(input_json) as o:
    params = json.load(o)

results = imputation_metrics(input_folder, input_file, algorithm, params)

results_file = f"hpo_results/{input_file}/{algorithm}/trials.json"
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
