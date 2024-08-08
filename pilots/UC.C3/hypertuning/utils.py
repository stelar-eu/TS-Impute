import pandas as pd
import time
import numpy as np
from stelarImputation import imputation as tsi
import optuna
from optuna.samplers import TPESampler
import os
from pypots.utils.metrics import calc_mae, calc_mre, calc_rmse, calc_mse
from sklearn.metrics import r2_score


def calc_vintage_score(data):
    data = data[(data.index.month >= 4) & (data.index.month <= 9)]

    data_Media = data.iloc[:, 0]
    data_Media = data_Media.resample('D').mean()

    data_Min = data.iloc[:, 2]
    data_Min = data_Min.resample('D').min()

    data_Prec = data.iloc[:, 3]
    data_Prec = data_Prec.resample('D').sum()

    data_aug_sep = data_Min[(data_Min.index.month >= 8) & (data_Min.index.month <= 9)].copy()
    data_jun_sep = data_Prec[(data_Prec.index.month >= 6) & (data_Prec.index.month <= 9)].copy()

    Tmean_apr_sep = data_Media.mean()
    Tmin_aug_sep = data_aug_sep.mean()
    prec_jun_sep = data_jun_sep.sum()

    vintage_score_dict = {'Tmean_apr_sep': Tmean_apr_sep, 'Tmin_aug_sep': Tmin_aug_sep, 'prec_jun_sep': prec_jun_sep}

    # Calculate the vintage score
    vintage_score = (22.38 * Tmean_apr_sep) - (0.6790 * (Tmean_apr_sep ** 2)) - (0.4089 * Tmin_aug_sep) - (
            0.006918 * prec_jun_sep) - 170.9
    vintage_score_dict['vintage_score'] = vintage_score
    return vintage_score_dict


def gap_generation_overlapping_perc(df: pd.DataFrame, perc: float, length: int, random_seed: int):
    gap_df = df.copy()

    assert 0 <= perc <= 1, 'Percentage %i is not between 0 and 1.' % perc
    assert 1 <= length <= 48, 'Length %i is not between 1 and 48.' % length

    mask = np.zeros((len(gap_df), len(gap_df.columns)), dtype=bool)
    rng = np.random.RandomState(random_seed)
    num = rng.randint(0, len(gap_df) - 48)
    start = num
    end = num + length

    start2 = None
    end2 = None

    if perc == 0:
        while (not start2):
            num = rng.randint(0, len(gap_df) - 48)
            if abs(num - start) > 48:
                start2 = num
                end2 = num + length

    else:
        start2 = start + int((1 - perc) * length)
        end2 = start2 + length

    mask[start:end, [0, 1, 2]] = True
    mask[start2:end2, 3] = True

    gap_df[mask] = np.nan

    return gap_df


def compute_metrics(original_df: pd.DataFrame, missing_df: pd.DataFrame, imputed_df: pd.DataFrame) -> dict:
    """
        Scale back the data to the original representation.
        :param original_df: original/ground truth dataframe.
        :param missing_df:  dataframe with the missing values.
        :param imputed_df: dataframe with the imputed values.
        :return: dictionary with the calculated metrics
        """
    mask = np.isnan(missing_df) ^ np.isnan(original_df)

    metrics = dict()
    original_df_np = np.nan_to_num(original_df.to_numpy())
    imputed_df_np = imputed_df.to_numpy()
    mask_np = mask.to_numpy()

    metrics['Missing value percentage'] = (missing_df.isna().sum().sum() * 100) / (
            missing_df.shape[0] * missing_df.shape[1])

    metrics['Missing value percentage between 4 and 9'] = (mask.sum().sum() * 100) / (
            missing_df[(missing_df.index.month >= 4) & (missing_df.index.month <= 9)].shape[0] *
            missing_df[(missing_df.index.month >= 4) & (missing_df.index.month <= 9)].shape[1])

    metrics['Mean absolute error'] = calc_mae(
        imputed_df_np,
        original_df_np,
        mask_np
    )

    metrics['Mean square error'] = calc_mse(
        imputed_df_np,
        original_df_np,
        mask_np
    )

    metrics['Root mean square error'] = calc_rmse(
        imputed_df_np,
        original_df_np,
        mask_np
    )

    metrics['Mean relative error'] = calc_mre(
        imputed_df_np,
        original_df_np,
        mask_np
    )

    metrics['Euclidean Distance'] = np.linalg.norm((imputed_df_np - original_df_np)[mask_np])

    # Flatten the arrays and apply the mask
    imputed_df_np_flat = imputed_df_np[mask_np].flatten()
    original_df_np_flat = original_df_np[mask_np].flatten()

    # Calculate the R-squared score
    metrics['r2 score'] = r2_score(original_df_np_flat, imputed_df_np_flat)

    return metrics


def imputation_metrics(input_folder: str, input_files: list, input_file: str, algorithm: str, params: dict, perc: float,
                       length: int):
    data_init = pd.DataFrame()

    for file in input_files:
        df = pd.read_excel(os.path.join(input_folder, file + '.xlsx'), sheet_name='Worksheet', index_col=0, skiprows=4,
                           skipfooter=3)
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')

        df.columns = ['Average T', 'Max T', 'Min T', 'Precipitation']
        df.columns = df.columns.map(lambda x: file + '_' + x)
        df = df[~df.index.duplicated()]
        if data_init.empty:
            data_init = df
        else:
            data_init = pd.concat([data_init, df], axis=1)

    algorithms = [algorithm]

    # data scaling
    data_init_scaled, scaler = tsi.dataframe_scaler(df_input=data_init, time_column='Time', preprocessing=False,
                                                    index=True)

    # add artificial missing values between 8 and 9 month
    data_missing = data_init_scaled.copy()

    data_missing.loc[
        (data_missing.index.month >= 8) & (data_missing.index.month <= 9), data_init.columns.str.startswith(
            input_file)] = gap_generation_overlapping_perc(data_missing.loc[(data_missing.index.month >= 8) & (
                data_missing.index.month <= 9), data_init.columns.str.startswith(input_file)], 0, 24, 2023)

    # hypertuning with optuna
    results = {}
    if algorithm.lower() == 'naive':
        # run imputation
        start_time = time.time()
        data_filled = tsi.run_imputation(missing=data_missing, algorithms=algorithms,
                                         time_column='Time', params={}, preprocessing=False,
                                         index=True, is_multivariate=True)[algorithms[0]]
        exec_time = (time.time() - start_time)

        data_filled.set_index('Time', inplace=True)

        # transform to original
        # missing
        data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                         scaler=scaler,
                                                         time_column='Time',
                                                         preprocessing=False,
                                                         index=True)
        # imputed
        data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                        scaler=scaler,
                                                        time_column='Time',
                                                        preprocessing=False,
                                                        index=True)

        # metrics
        names = data_init.columns.str.startswith(input_file)

        metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names], data_filled.loc[:, names])
        metrics['Execution Time'] = exec_time

        results['result'] = {}
        results['result']['metrics'] = metrics

        # vintage score
        vintage_score = {}
        vintage_init = calc_vintage_score(data_init.loc[:, names])
        vintage_score['original'] = vintage_init

        vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
        vintage_score['missing'] = vintage_missing

        vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
        vintage_score['filled'] = vintage_filled

        # vintage score deviation
        percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                            vintage_init['vintage_score']) * 100)
        percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                           vintage_init['vintage_score']) * 100)

        vintage_score['Deviation missing'] = percentage_deviation_missing
        vintage_score['Deviation filled'] = percentage_deviation_filled
        vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

        results['result']['vintage_score'] = vintage_score

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
                                                 time_column='Time', params=parameters, preprocessing=False,
                                                 index=True, is_multivariate=True)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index('Time', inplace=True)
                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column='Time',
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column='Time',
                                                                preprocessing=False,
                                                                index=True)

                # metrics
                names = data_init.columns.str.startswith(input_file)

                metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names],
                                          data_filled.loc[:, names])
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # vintage score
                vintage_score = {}
                vintage_init = calc_vintage_score(data_init.loc[:, names])
                vintage_score['original'] = vintage_init

                vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
                vintage_score['missing'] = vintage_missing

                vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
                vintage_score['filled'] = vintage_filled

                # vintage score deviation
                percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                                    vintage_init['vintage_score']) * 100)
                percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                                   vintage_init['vintage_score']) * 100)

                vintage_score['Deviation missing'] = percentage_deviation_missing
                vintage_score['Deviation filled'] = percentage_deviation_filled
                vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

                results[trial.number]['result']['vintage_score'] = vintage_score
                if metrics['Mean square error'] > 100 or np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                results[trial.number]['result'] = 'An exception occured, skip trial!'
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
                                                 time_column='Time', params=parameters, preprocessing=False,
                                                 index=True, is_multivariate=True)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index('Time', inplace=True)

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column='Time',
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column='Time',
                                                                preprocessing=False,
                                                                index=True)

                # metrics
                names = data_init.columns.str.startswith(input_file)

                metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names],
                                          data_filled.loc[:, names])
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # vintage score
                vintage_score = {}
                vintage_init = calc_vintage_score(data_init.loc[:, names])
                vintage_score['original'] = vintage_init

                vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
                vintage_score['missing'] = vintage_missing

                vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
                vintage_score['filled'] = vintage_filled

                # vintage score deviation
                percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                                    vintage_init['vintage_score']) * 100)
                percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                                   vintage_init['vintage_score']) * 100)

                vintage_score['Deviation missing'] = percentage_deviation_missing
                vintage_score['Deviation filled'] = percentage_deviation_filled
                vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

                results[trial.number]['result']['vintage_score'] = vintage_score
                if metrics['Mean square error'] > 100 or np.isnan(metrics['Mean square error']):
                    already_seen.remove(max_rank)
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                results[trial.number]['result'] = 'An exception occured, skip trial!'
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
                                                 time_column='Time', params=parameters, preprocessing=False,
                                                 index=True, is_multivariate=True)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index('Time', inplace=True)

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column='Time',
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column='Time',
                                                                preprocessing=False,
                                                                index=True)

                # metrics
                names = data_init.columns.str.startswith(input_file)

                metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names],
                                          data_filled.loc[:, names])
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # vintage score
                vintage_score = {}
                vintage_init = calc_vintage_score(data_init.loc[:, names])
                vintage_score['original'] = vintage_init

                vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
                vintage_score['missing'] = vintage_missing

                vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
                vintage_score['filled'] = vintage_filled

                # vintage score deviation
                percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                                    vintage_init['vintage_score']) * 100)
                percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                                   vintage_init['vintage_score']) * 100)

                vintage_score['Deviation missing'] = percentage_deviation_missing
                vintage_score['Deviation filled'] = percentage_deviation_filled
                vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

                results[trial.number]['result']['vintage_score'] = vintage_score

                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                results[trial.number]['result'] = 'An exception occured, skip trial!'
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
                                                 time_column='Time', params=parameters, preprocessing=False,
                                                 index=True, is_multivariate=True)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index('Time', inplace=True)

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column='Time',
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column='Time',
                                                                preprocessing=False,
                                                                index=True)

                # metrics
                names = data_init.columns.str.startswith(input_file)

                metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names],
                                          data_filled.loc[:, names])
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # vintage score
                vintage_score = {}
                vintage_init = calc_vintage_score(data_init.loc[:, names])
                vintage_score['original'] = vintage_init

                vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
                vintage_score['missing'] = vintage_missing

                vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
                vintage_score['filled'] = vintage_filled

                # vintage score deviation
                percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                                    vintage_init['vintage_score']) * 100)
                percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                                   vintage_init['vintage_score']) * 100)

                vintage_score['Deviation missing'] = percentage_deviation_missing
                vintage_score['Deviation filled'] = percentage_deviation_filled
                vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

                results[trial.number]['result']['vintage_score'] = vintage_score

                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                results[trial.number]['result'] = 'An exception occured, skip trial!'
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
                                                 time_column='Time', params=parameters, preprocessing=False,
                                                 index=True, is_multivariate=True)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index('Time', inplace=True)

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column='Time',
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column='Time',
                                                                preprocessing=False,
                                                                index=True)

                # metrics
                names = data_init.columns.str.startswith(input_file)

                # metrics
                metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names],
                                          data_filled.loc[:, names])
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # vintage score
                vintage_score = {}
                vintage_init = calc_vintage_score(data_init.loc[:, names])
                vintage_score['original'] = vintage_init

                vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
                vintage_score['missing'] = vintage_missing

                vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
                vintage_score['filled'] = vintage_filled

                # vintage score deviation
                percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                                    vintage_init['vintage_score']) * 100)
                percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                                   vintage_init['vintage_score']) * 100)

                vintage_score['Deviation missing'] = percentage_deviation_missing
                vintage_score['Deviation filled'] = percentage_deviation_filled
                vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

                results[trial.number]['result']['vintage_score'] = vintage_score

                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                results[trial.number]['result'] = 'An exception occured, skip trial!'
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
                                                 time_column='Time', params=parameters, preprocessing=False,
                                                 index=True, is_multivariate=True)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index('Time', inplace=True)

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column='Time',
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column='Time',
                                                                preprocessing=False,
                                                                index=True)

                # metrics
                names = data_init.columns.str.startswith(input_file)

                metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names],
                                          data_filled.loc[:, names])
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # vintage score
                vintage_score = {}
                vintage_init = calc_vintage_score(data_init.loc[:, names])
                vintage_score['original'] = vintage_init

                vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
                vintage_score['missing'] = vintage_missing

                vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
                vintage_score['filled'] = vintage_filled

                # vintage score deviation
                percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                                    vintage_init['vintage_score']) * 100)
                percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                                   vintage_init['vintage_score']) * 100)

                vintage_score['Deviation missing'] = percentage_deviation_missing
                vintage_score['Deviation filled'] = percentage_deviation_filled
                vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

                results[trial.number]['result']['vintage_score'] = vintage_score

                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                results[trial.number]['result'] = 'An exception occured, skip trial!'
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
                                                 time_column='Time', params=parameters, preprocessing=False,
                                                 index=True, is_multivariate=True)[algorithms[0]]
                exec_time = (time.time() - start_time)

                data_filled.set_index('Time', inplace=True)

                # transform to original
                # missing
                data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                                 scaler=scaler,
                                                                 time_column='Time',
                                                                 preprocessing=False,
                                                                 index=True)
                # imputed
                data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                                scaler=scaler,
                                                                time_column='Time',
                                                                preprocessing=False,
                                                                index=True)

                # metrics
                names = data_init.columns.str.startswith(input_file)

                metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names],
                                          data_filled.loc[:, names])
                metrics['Execution Time'] = exec_time

                results[trial.number]['result'] = {}
                results[trial.number]['result']['metrics'] = metrics

                # vintage score
                vintage_score = {}
                vintage_init = calc_vintage_score(data_init.loc[:, names])
                vintage_score['original'] = vintage_init

                vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
                vintage_score['missing'] = vintage_missing

                vintage_filled = calc_vintage_score(data_filled_orig.loc[:, names])
                vintage_score['filled'] = vintage_filled

                # vintage score deviation
                percentage_deviation_missing = abs(((vintage_missing['vintage_score'] - vintage_init['vintage_score']) /
                                                    vintage_init['vintage_score']) * 100)
                percentage_deviation_filled = abs(((vintage_filled['vintage_score'] - vintage_init['vintage_score']) /
                                                   vintage_init['vintage_score']) * 100)

                vintage_score['Deviation missing'] = percentage_deviation_missing
                vintage_score['Deviation filled'] = percentage_deviation_filled
                vintage_score['Improvement'] = (percentage_deviation_missing - percentage_deviation_filled)

                results[trial.number]['result']['vintage_score'] = vintage_score

                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                results[trial.number]['result'] = 'An exception occured, skip trial!'
                return 100

    if algorithm.lower() != 'naive':
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=2023))
        study.optimize(objective, n_trials=50)

        results['best_trial'] = study.best_trial.number

    return results
