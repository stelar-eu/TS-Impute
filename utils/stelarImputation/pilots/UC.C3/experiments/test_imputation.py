from stelarImputation import imputation as tsi
from utils import test_imputation_metrics
import json 
import optuna
from optuna.samplers import TPESampler
import os
import sys
import numpy as np
import time

algorithm = sys.argv[1] # 'SoftImpute'
input_folder = str(sys.argv[2])
input_file = str(sys.argv[3]) # 'Report(Torrevilla2023)'
length = int(sys.argv[4]) # '120'
perc = float(sys.argv[5]) # '0'
seed = int(sys.argv[6]) # '2023'
input_json = sys.argv[7] # 'SoftImpute'
input_files = list(sys.argv[8:])


with open(input_json) as o:
    best_params = json.load(o)

params = best_params[input_file]

if algorithm in ["SoftImpute", "CDMissingValueRecovery"]:
    result = 20
    while result >= 20 or np.isnan(result):
        results = test_imputation_metrics(input_folder, input_files, input_file, algorithm, params, perc, length, seed)
        result = results['metrics']['Mean square error']
        print(result)
        time.sleep(1)
else:
    results = test_imputation_metrics(input_folder, input_files, input_file, algorithm, params, perc, length, seed)
    result = results['metrics']['Mean square error']
    print(result)

results_file = f"experiment_results/{input_file}/{algorithm}/seed={seed}_length={length}_perc={perc}.json"                    
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