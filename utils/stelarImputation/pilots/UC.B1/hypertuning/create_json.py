import json
import os
import sys

datasets = sys.argv[1].split(',')
algorithms = sys.argv[2].split(',')

hypertuning_results = {}
for dataset in datasets:
    hypertuning_results[dataset] = {}
    for algorithm in algorithms:
        if algorithm.lower() != 'naive':
            hypertuning_results[dataset][algorithm] = {}
            results_file = f"hpo_results/{dataset}/{algorithm}/trials.json"

            with open(results_file) as o:
                params = json.load(o)

            hypertuning_results[dataset][algorithm] = params[str(params['best_trial'])]['parameters'][algorithm]

save_file = '../experiments/hypertuning_results.json'

if os.path.exists(save_file):
    os.remove(save_file)
else:
    print("The file does not exist")

json_object = json.dumps(hypertuning_results, indent=4)

# Writing to sample.json
with open(save_file, "w") as outfile:
    outfile.write(json_object)
