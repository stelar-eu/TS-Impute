from stelarImputation import imputation as tsi
from utils import imputation_metrics
import json
import optuna
from optuna.samplers import TPESampler
import os
import sys
import ast

input_folder = str(sys.argv[1])  # '../../data'
input_file = str(sys.argv[2])  # 'Report(Torrevilla2023)'
perc = float(sys.argv[3])  # 0
length = int(sys.argv[4])  # 24
algorithm = str(sys.argv[5])  # 'SoftImpute'
input_json = str(sys.argv[6])  # 'input.json'
input_files = list(sys.argv[7:])  # ['Report(Torrevilla2023)', 'Report(Barbieri2023)', 'Report(Bertelegni2023)', 'Report(Rebutti2023)']

with open(input_json) as o:
    params = json.load(o)

results = imputation_metrics(input_folder, input_files, input_file, algorithm, params, perc, length)

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
