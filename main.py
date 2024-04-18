import math
import pandas as pd
import numpy as np
from stelarImputation import imputation as tsi
from typing import Union
from minio import Minio
import json
import uuid
import sys
from xgboost import XGBRegressor
from pathlib import Path
import tempfile
import time
import ast


def prep_df(input_file, header, sep, minio):
    """
    Prepare DataFrame from input file.
    """
    if input_file.startswith('s3://'):
        bucket, key = input_file.replace('s3://', '').split('/', 1)
        client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
        df = pd.read_csv(client.get_object(bucket, key), header=header, sep=sep)
    else:
        df = pd.read_csv(input_file, header=header, sep=sep)

    return df


def prep_model(input_file, minio):
    """
    Prepare DataFrame from input file.
    """
    model_xgb = XGBRegressor()
    if input_file.startswith('s3://'):
        bucket, key = input_file.replace('s3://', '').split('/', 1)
        client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
        dict_as_string = client.get_object(bucket, key).data.decode("utf-8")
        # Create a temporary file (in-memory) with a .json extension
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+') as temp_file:
            # Write the JSON data to the temporary file
            temp_file.write(dict_as_string)
            temp_file.seek(0)

            model_xgb.load_model(temp_file.name)
    else:
        model_xgb.load_model(input_file)
    return model_xgb


def find_type(dict_or_list_str: Union[str, dict, list]):
    """
    Check if the dictionary or list is of str type. 
    If we get a string then evaluate it and get the Python dictionary or list object.
    """
    if type(dict_or_list_str) == str:
        return ast.literal_eval(dict_or_list_str)
    else:
        return dict_or_list_str


def run(j):
    try:
        inputs = j['input']

        parameters = j['parameters']

        if 'imp_type' in parameters:
            imp_type = parameters['imp_type']
            if imp_type not in ['gap_generation', 'imputation', 'train_ensemble', 'imputation_ensemble']:
                raise ValueError(
                    "imp_type must be in ['gap_generation', 'imputation', 'train_ensemble', 'imputation_ensemble']")
        else:
            raise ValueError(
                "Missing imp_type. It must be in ['gap_generation', 'imputation', 'train_ensemble', 'imputation_ensemble']")

        if 'time_column' in parameters:
            time_column = parameters['time_column']
        else:
            time_column = ''

        if 'datetime_format' in parameters:
            datetime_format = parameters['datetime_format']
        else:
            datetime_format = ''

        if 'spatial_x_column' in parameters:
            spatial_x_column = parameters['spatial_x_column']
        else:
            spatial_x_column = ''

        if 'spatial_y_column' in parameters:
            spatial_y_column = parameters['spatial_y_column']
        else:
            spatial_y_column = ''

        header = parameters['header']
        sep = parameters['sep']
        
        if 'dimension_column' in parameters:
            dimension_column = parameters['dimension_column']
        else:
            dimension_column = ''
            
        

        if 'is_multivariate' in parameters:
            is_multivariate = parameters['is_multivariate']
        else:
            is_multivariate = False

        if 'areaVStime' in parameters:
            areaVStime = parameters['areaVStime']
        else:
            areaVStime = 0

        if 'algorithms' in parameters:
            algorithms = find_type(parameters['algorithms'])
        else:
            algorithms = dict()

        if 'params' in parameters:
            params = find_type(parameters['params'])
        else:
            params = dict()

        if 'preprocessing' in parameters:
            preprocessing = parameters['preprocessing']
        else:
            preprocessing = True

        if 'index' in parameters:
            index = parameters['index']
        else:
            index = False

        output_file = parameters['output_file']
        minio = j['minio']

        if imp_type == 'gap_generation':
            ground_truth = inputs[0]
            train_params = find_type(parameters['train_params'])
            df = prep_df(ground_truth, header, sep, minio)
            start_time = time.time()

            df_gaps = tsi.run_gap_generation(ground_truth=df,
                                             train_params=train_params,
                                             dimension_column=dimension_column,
                                             time_column=time_column,
                                             datetime_format=datetime_format,
                                             spatial_x_column=spatial_x_column,
                                             spatial_y_column=spatial_y_column,
                                             preprocessing=preprocessing,
                                             index=index)

            log = {'total_time': (time.time() - start_time), 'missing values': str(df_gaps.isnull().sum().sum())}
            output_split = output_file.split('.')

            # include gap type and parameters in the file name
            added_text = ''
            gap_type = train_params['gap_type']
            if gap_type == 'random':
                max_gap_length = train_params['max_gap_length']
                max_gap_count = train_params['max_gap_count']

                added_text = ('_random_max_gap_length=' + str(max_gap_length) +
                              '_max_gap_count=' + str(max_gap_count))
            elif gap_type == 'single':
                miss_perc = train_params['miss_perc']
                added_text = '_single_miss_perc=' + str(miss_perc)
            elif gap_type == 'blackout':
                gap_length = train_params['gap_length']
                added_text = '_blackout_gap_length=' + str(gap_length)
            else:
                added_text = '_no_overlap'

            name = str('.'.join(output_split[:-1]) + added_text)

            df_gaps.to_csv(output_file, index=False, sep=',', header=True)

            # Save to minio
            basename = str(uuid.uuid4()) + "." + output_file.split('.')[-1]
            client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
            result = client.fput_object(minio['bucket'], basename, output_file)
            object_path = f"s3://{result.bucket_name}/{result.object_name}"

            return {'message': 'StelarImputation project Gap Generation executed successfully!',
                    'output': [{"name": name, "path": object_path}],
                    'metrics': log,
                    'status': 200}

        elif imp_type == 'imputation':
            imputation_input = inputs[0]
            df = prep_df(imputation_input, header, sep, minio)
            start_time = time.time()

            dict_of_imputed_dfs = tsi.run_imputation(missing=df,
                                                     algorithms=algorithms, params=params,
                                                     dimension_column=dimension_column,
                                                     time_column=time_column,
                                                     datetime_format=datetime_format,
                                                     spatial_x_column=spatial_x_column,
                                                     spatial_y_column=spatial_y_column,
                                                     is_multivariate=is_multivariate,
                                                     areaVStime=areaVStime,
                                                     preprocessing=preprocessing,
                                                     index=index)

            log = {'total_time': (time.time() - start_time)}

            # save csvs + execution_time metrics
            output = []
            for imputed_df_name in dict_of_imputed_dfs:
                output_split = output_file.split('.')
                alg_output_file = str('.'.join(output_split[:-1]) + '_' + imputed_df_name) + '.' + output_split[-1]
                dict_of_imputed_dfs[imputed_df_name].to_csv(alg_output_file, index=False, sep=',', header=True)

                # Save to minio
                basename = str(uuid.uuid4()) + "." + output_split[-1]
                client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
                result = client.fput_object(minio['bucket'], basename, alg_output_file)
                object_path = f"s3://{result.bucket_name}/{result.object_name}"
                output.append({"name": imputed_df_name, "path": object_path})

            return {'message': 'StelarImputation project Imputation executed successfully!',
                    'output': output,
                    'metrics': log,
                    'status': 200}

        elif imp_type == 'train_ensemble':
            ground_truth = inputs[0]
            train_params = find_type(parameters['train_params'])
            df = prep_df(ground_truth, header, sep, minio)

            start_time = time.time()
            model, metrics = tsi.train_ensemble(ground_truth=df,
                                                algorithms=algorithms,
                                                params=params, train_params=train_params,
                                                dimension_column=dimension_column,
                                                time_column=time_column,
                                                datetime_format=datetime_format,
                                                spatial_x_column=spatial_x_column,
                                                spatial_y_column=spatial_y_column,
                                                is_multivariate=is_multivariate,
                                                areaVStime=areaVStime,
                                                preprocessing=preprocessing,
                                                index=index)

            log = {'total_time': (time.time() - start_time)}
            log.update(metrics)
            # Save model
            output_split = output_file.split('.')
            model_output_file = str('.'.join(output_split[:-1]) + '_imputation_model.json')
            model.save_model(model_output_file)

            # Save to minio
            basename = str(uuid.uuid4()) + "." + model_output_file.split('.')[-1]
            client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
            result = client.fput_object(minio['bucket'], basename, model_output_file)
            object_path = f"s3://{result.bucket_name}/{result.object_name}"

            return {'message': 'StelarImputation project Train Ensemble Model executed successfully!',
                    'output': [{"name": 'Model in JSON format', "path": object_path}],
                    'metrics': log,
                    'status': 200}

        elif imp_type == 'imputation_ensemble':
            imputation_input = inputs[0]
            model_input = inputs[1]
            df = prep_df(imputation_input, header, sep, minio)
            model = prep_model(model_input, minio)

            start_time = time.time()
            imputed_df = tsi.run_imputation_ensemble(missing=df,
                                                     algorithms=algorithms,
                                                     params=params, model=model,
                                                     dimension_column=dimension_column,
                                                     time_column=time_column,
                                                     datetime_format=datetime_format,
                                                     spatial_x_column=spatial_x_column,
                                                     spatial_y_column=spatial_y_column,
                                                     is_multivariate=is_multivariate,
                                                     areaVStime=areaVStime,
                                                     preprocessing=preprocessing,
                                                     index=index)

            log = {'total_time': (time.time() - start_time)}
            imputed_df.to_csv(output_file, index=False, sep=',', header=True)

            # Save to minio
            basename = str(uuid.uuid4()) + "." + output_file.split('.')[-1]
            client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
            result = client.fput_object(minio['bucket'], basename, output_file)
            object_path = f"s3://{result.bucket_name}/{result.object_name}"

        return {'message': 'StelarImputation project Imputation Ensemble Model executed successfully!',
                'output': [{"name": 'Model imputed results in CSV format', "path": object_path}],
                'metrics': log,
                'status': 200}
    except Exception as e:
        return {
            'message': 'An error occurred during data processing.',
            'error': str(e),
            'status': 500
        }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))
