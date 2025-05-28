import json
import os
import sys

datasets = sys.argv[1].split(',')
algorithms = sys.argv[2].split(',')

hypertuning_results = {}
for dataset in datasets:
    original_dataset = dataset.split('sample_')[1]
    hypertuning_results[original_dataset] = {}
    for algorithm in algorithms:
        if algorithm.lower() != 'naive':
            hypertuning_results[original_dataset][algorithm] = {}
            results_file = f"hpo_results/{dataset}/{algorithm}/trials.json"

            with open(results_file) as o:
                params = json.load(o)
            best = params[str(params['best_trial'])]['parameters'][algorithm]

            hypertuning_results[original_dataset][algorithm] = best

save_file = '../experiments/hypertuning_results.json'

if os.path.exists(save_file):
    os.remove(save_file)
else:
    print("The file does not exist")

json_object = json.dumps(hypertuning_results, indent=4)

# Writing to sample.json
with open(save_file, "w") as outfile:
    outfile.write(json_object)
