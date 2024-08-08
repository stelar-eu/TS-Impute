import pandas as pd
import time
import numpy as np
from stelarImputation import imputation as tsi
import optuna
from optuna.samplers import TPESampler
import os


def custom_single_gap(df: pd.DataFrame, random_state: int = 2024):
    df_miss = df.copy()
    state = np.random.RandomState(random_state)
    non_nan_idx = [np.where(~pd.isna(df_miss[col]))[0] for col in df.columns]
    random_rows = [state.choice(idx) for idx in non_nan_idx]
    mask = np.zeros(df_miss.shape, dtype=bool)

    mask[random_rows, np.arange(df_miss.shape[1])] = True

    df_miss.values[mask] = np.nan
    return df_miss


def imputation_metrics(input_folder: str, input_file: str, algorithm: str, params: dict):
    file_path = os.path.join(input_folder, (input_file + '.csv'))
    algorithms = [algorithm]

    # read file
    data_init = pd.read_csv(file_path, index_col=0).sort_index().dropna(axis=0, how='all')

    # data scaling
    data_init_scaled, scaler = tsi.dataframe_scaler(df_input=data_init, time_column=data_init.index.name,
                                                    preprocessing=False,
                                                    index=True)

    # add artificial missing values
    data_missing = data_init_scaled.copy()
    data_missing = custom_single_gap(data_missing)

    # hypertuning with optuna
    results = {}
    if algorithm.lower() == 'naive':
        # run imputation
        start_time = time.time()
        data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                         time_column=data_init.index.name, params={}, preprocessing=False,
                                         index=True, is_multivariate=False)[algorithms[0]]
        exec_time = (time.time() - start_time)

        data_filled.set_index(data_init.index.name, inplace=True)

        # metrics
        metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
        metrics['Execution Time'] = exec_time

        results['result'] = {}
        results['result']['metrics'] = metrics

        # transform to original
        # missing
        data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                         scaler=scaler,
                                                         time_column=data_init.index.name,
                                                         preprocessing=False,
                                                         index=True)
        # imputed
        data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                        scaler=scaler,
                                                        time_column=data_init.index.name,
                                                        preprocessing=False,
                                                        index=True)
        
    elif algorithm.lower() == 'cdmissingvaluerecovery':
        model_parameters = params[algorithm]

        def objective(trial):
            truncation = trial.suggest_categorical("truncation", model_parameters['truncation'])
            eps = trial.suggest_float("eps", model_parameters['eps'][0], model_parameters['eps'][1])
            parameters = {
                algorithm: {
                    "truncation": truncation,
                    "eps": eps
                }
            }
            results[trial.number] = {}
            results[trial.number]['parameters'] = parameters

            try:
                # run imputation
                start_time = time.time()
                data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                                 time_column=data_init.index.name, params=parameters,
                                                 preprocessing=False,
                                                 index=True, is_multivariate=False)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index(data_init.index.name, inplace=True)

                # metrics
                metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column=data_init.index.name,
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column=data_init.index.name,
                                                                preprocessing=False,
                                                                index=True)

                if metrics['Mean square error'] > 100 or np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occurred, skip trial!')
                results[trial.number]['result'] = 'An exception occurred, skip trial!'
                return 100

    elif algorithm.lower() == 'softimpute':
        model_parameters = params[algorithm]
        already_seen = []

        def objective(trial):

            max_rank = trial.suggest_categorical("max_rank", model_parameters['max_rank'])

            parameters = {
                algorithm: {
                    "max_rank": max_rank
                }
            }

            if max_rank in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(max_rank)

            results[trial.number] = {}
            results[trial.number]['parameters'] = parameters

            try:
                # run imputation
                start_time = time.time()
                data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                                 time_column=data_init.index.name, params=parameters,
                                                 preprocessing=False,
                                                 index=True, is_multivariate=False)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index(data_init.index.name, inplace=True)

                # metrics
                metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column=data_init.index.name,
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column=data_init.index.name,
                                                                preprocessing=False,
                                                                index=True)

                if metrics['Mean square error'] > 100 or np.isnan(metrics['Mean square error']):
                    already_seen.remove(max_rank)
                    raise optuna.exceptions.TrialPruned()

                return metrics['Mean square error']

            except:
                print('An exception occurred, skip trial!')
                results[trial.number]['result'] = 'An exception occurred, skip trial!'
                return 100

    elif algorithm.lower() == 'dynammo':
        model_parameters = params[algorithm]
        already_seen = []

        def objective(trial):

            H = trial.suggest_int("H", model_parameters['H'][0], model_parameters['H'][1])
            max_iter = trial.suggest_categorical("max_iter", model_parameters['max_iter'])
            fast = trial.suggest_categorical("fast", model_parameters['fast'])

            parameters = {
                algorithm: {
                    "H": H,
                    "max_iter": max_iter,
                    "fast": fast
                }
            }
            value = str(H) + '_' + str(fast)
            if value in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(value)

            results[trial.number] = {}
            results[trial.number]['parameters'] = parameters

            try:
                # run imputation
                start_time = time.time()
                data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                                 time_column=data_init.index.name, params=parameters,
                                                 preprocessing=False,
                                                 index=True, is_multivariate=False)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index(data_init.index.name, inplace=True)

                # metrics
                metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column=data_init.index.name,
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column=data_init.index.name,
                                                                preprocessing=False,
                                                                index=True)

                return metrics['Mean square error']

            except:
                print('An exception occurred, skip trial!')
                results[trial.number]['result'] = 'An exception occurred, skip trial!'
                return 100

    elif algorithm.lower() == 'saits':
        model_parameters = params[algorithm]

        def objective(trial):

            n_layers = trial.suggest_categorical("n_layers", model_parameters['n_layers'])
            d_ffn = trial.suggest_categorical("d_ffn", model_parameters['d_ffn'])
            n_heads = trial.suggest_categorical("n_heads", model_parameters['n_heads'])
            d_k = trial.suggest_categorical("d_k", model_parameters['d_k'])
            d_v = trial.suggest_categorical("d_v", model_parameters['d_v'])
            dropout = trial.suggest_categorical("dropout", model_parameters['dropout'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            ORT_weight = trial.suggest_categorical("ORT_weight", model_parameters['ORT_weight'])
            MIT_weight = trial.suggest_categorical("MIT_weight", model_parameters['MIT_weight'])
            attn_dropout = trial.suggest_categorical("attn_dropout", model_parameters['attn_dropout'])
            diagonal_attention_mask = trial.suggest_categorical("diagonal_attention_mask",
                                                                model_parameters['diagonal_attention_mask'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            device = trial.suggest_categorical("device", model_parameters['device'])
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)

            parameters = {
                algorithm: {
                    "n_layers": n_layers,
                    "d_model": n_heads * d_k,
                    "d_ffn": d_ffn,
                    "n_heads": n_heads,
                    "d_k": d_k,
                    "d_v": d_v,
                    "dropout": dropout,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "ORT_weight": ORT_weight,
                    "MIT_weight": MIT_weight,
                    "attn_dropout": attn_dropout,
                    "diagonal_attention_mask": diagonal_attention_mask,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "device": device
                }
            }

            results[trial.number] = {}
            results[trial.number]['parameters'] = parameters

            try:
                # run imputation
                start_time = time.time()
                data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                                 time_column=data_init.index.name, params=parameters,
                                                 preprocessing=False,
                                                 index=True, is_multivariate=False)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index(data_init.index.name, inplace=True)

                # metrics
                metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column=data_init.index.name,
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column=data_init.index.name,
                                                                preprocessing=False,
                                                                index=True)

                return metrics['Mean square error']

            except:
                print('An exception occurred, skip trial!')
                results[trial.number]['result'] = 'An exception occurred, skip trial!'
                return 100

    elif algorithm.lower() == 'timesnet':
        model_parameters = params[algorithm]

        def objective(trial):

            n_layers = trial.suggest_categorical("n_layers", model_parameters['n_layers'])
            d_model = trial.suggest_categorical("d_model", model_parameters['d_model'])
            d_ffn = trial.suggest_categorical("d_ffn", model_parameters['d_ffn'])
            n_heads = trial.suggest_categorical("n_heads", model_parameters['n_heads'])
            top_k = trial.suggest_categorical("top_k", model_parameters['top_k'])
            n_kernels = trial.suggest_categorical("n_kernels", model_parameters['n_kernels'])
            dropout = trial.suggest_categorical("dropout", model_parameters['dropout'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            attn_dropout = trial.suggest_categorical("attn_dropout", model_parameters['attn_dropout'])
            apply_nonstationary_norm = trial.suggest_categorical("apply_nonstationary_norm",
                                                                 model_parameters['apply_nonstationary_norm'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            device = trial.suggest_categorical("device", model_parameters['device'])
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)

            parameters = {
                algorithm: {
                    "n_layers": n_layers,
                    "d_model": d_model,
                    "d_ffn": d_ffn,
                    "n_heads": n_heads,
                    "top_k": top_k,
                    "n_kernels": n_kernels,
                    "dropout": dropout,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "attn_dropout": attn_dropout,
                    "apply_nonstationary_norm": apply_nonstationary_norm,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "device": device
                }
            }

            results[trial.number] = {}
            results[trial.number]['parameters'] = parameters

            try:
                # run imputation
                start_time = time.time()
                data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                                 time_column=data_init.index.name, params=parameters,
                                                 preprocessing=False,
                                                 index=True, is_multivariate=False)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index(data_init.index.name, inplace=True)

                # metrics
                metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column=data_init.index.name,
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column=data_init.index.name,
                                                                preprocessing=False,
                                                                index=True)

                return metrics['Mean square error']

            except:
                print('An exception occurred, skip trial!')
                results[trial.number]['result'] = 'An exception occurred, skip trial!'
                return 100

    elif algorithm.lower() == 'nonstationary_transformer':
        model_parameters = params[algorithm]

        def objective(trial):

            n_layers = trial.suggest_categorical("n_layers", model_parameters['n_layers'])
            d_model = trial.suggest_categorical("d_model", model_parameters['d_model'])
            d_ffn = trial.suggest_categorical("d_ffn", model_parameters['d_ffn'])
            n_heads = trial.suggest_categorical("n_heads", model_parameters['n_heads'])
            choices_tuple = tuple(tuple(choice) for choice in model_parameters['d_projector_hidden'])
            d_projector_hidden = trial.suggest_categorical("d_projector_hidden", choices_tuple)
            n_projector_hidden_layers = trial.suggest_categorical("n_projector_hidden_layers",
                                                                  model_parameters['n_projector_hidden_layers'])
            dropout = trial.suggest_categorical("dropout", model_parameters['dropout'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            ORT_weight = trial.suggest_categorical("ORT_weight", model_parameters['ORT_weight'])
            MIT_weight = trial.suggest_categorical("MIT_weight", model_parameters['MIT_weight'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            device = trial.suggest_categorical("device", model_parameters['device'])
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            print(list(d_projector_hidden))
            parameters = {
                algorithm: {
                    "n_layers": n_layers,
                    "d_model": d_model,
                    "d_ffn": d_ffn,
                    "n_heads": n_heads,
                    "n_projector_hidden_layers": n_projector_hidden_layers,
                    "d_projector_hidden": list(d_projector_hidden),
                    "dropout": dropout,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "ORT_weight": ORT_weight,
                    "MIT_weight": MIT_weight,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "device": device
                }
            }

            results[trial.number] = {}
            results[trial.number]['parameters'] = parameters

            try:
                # run imputation
                start_time = time.time()
                data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                                 time_column=data_init.index.name, params=parameters,
                                                 preprocessing=False,
                                                 index=True, is_multivariate=False)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index(data_init.index.name, inplace=True)

                # metrics
                metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column=data_init.index.name,
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column=data_init.index.name,
                                                                preprocessing=False,
                                                                index=True)
                return metrics['Mean square error']

            except:
                print('An exception occurred, skip trial!')
                results[trial.number]['result'] = 'An exception occurred, skip trial!'
                return 100

    elif algorithm.lower() == 'autoformer':
        model_parameters = params[algorithm]

        def objective(trial):

            n_layers = trial.suggest_categorical("n_layers", model_parameters['n_layers'])
            d_model = trial.suggest_categorical("d_model", model_parameters['d_model'])
            d_ffn = trial.suggest_categorical("d_ffn", model_parameters['d_ffn'])
            n_heads = trial.suggest_categorical("n_heads", model_parameters['n_heads'])
            factor = trial.suggest_categorical("factor", model_parameters['factor'])
            moving_avg_window_size = trial.suggest_categorical("moving_avg_window_size",
                                                               model_parameters['moving_avg_window_size'])
            dropout = trial.suggest_categorical("dropout", model_parameters['dropout'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            ORT_weight = trial.suggest_categorical("ORT_weight", model_parameters['ORT_weight'])
            MIT_weight = trial.suggest_categorical("MIT_weight", model_parameters['MIT_weight'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            device = trial.suggest_categorical("device", model_parameters['device'])
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)

            parameters = {
                algorithm: {
                    "n_layers": n_layers,
                    "d_model": d_model,
                    "d_ffn": d_ffn,
                    "n_heads": n_heads,
                    "factor": factor,
                    "moving_avg_window_size": moving_avg_window_size,
                    "dropout": dropout,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "ORT_weight": ORT_weight,
                    "MIT_weight": MIT_weight,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "device": device
                }
            }

            results[trial.number] = {}
            results[trial.number]['parameters'] = parameters

            try:
                # run imputation
                start_time = time.time()
                data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                                 time_column=data_init.index.name, params=parameters,
                                                 preprocessing=False,
                                                 index=True, is_multivariate=False)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index(data_init.index.name, inplace=True)

                # metrics
                metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column=data_init.index.name,
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column=data_init.index.name,
                                                                preprocessing=False,
                                                                index=True)

                return metrics['Mean square error']

            except:
                print('An exception occurred, skip trial!')
                results[trial.number]['result'] = 'An exception occurred, skip trial!'
                return 100

    if algorithm.lower() != 'naive':
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=2023))
        study.optimize(objective, n_trials=50)

        results['best_trial'] = study.best_trial.number

    return results
