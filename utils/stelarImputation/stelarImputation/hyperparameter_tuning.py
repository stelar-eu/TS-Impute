import numpy as np
import optuna
import pandas as pd
import time
from optuna.samplers import TPESampler
from torch import cuda
from stelarImputation.imputation import compute_metrics, run_gap_generation, run_imputation

__all__ = ['perform_hyperparameter_tuning']


def perform_hyperparameter_tuning(missing_df: pd.DataFrame, groundTruth_df: pd.DataFrame = None,
                                  algorithm: str = 'Naive', params: dict = None,
                                  n_trials: int = 20, time_column: str = '', dimension_column: str = '',
                                  spatial_x_column: str = '',
                                  spatial_y_column: str = '',
                                  preprocessing: bool = False, index: bool = False,
                                  is_multivariate: bool = False, areaVStime: int = 0):
    """
    Perform hyperparameter tuning for a specified imputation algorithm using Optuna.
    :param missing_df: DataFrame containing missing values to be imputed.
    :param groundTruth_df: Optional DataFrame containing the ground truth values for evaluation.
    :param algorithm: The name of the imputation algorithm to tune.
    :param params: Dictionary of parameters for the specified algorithm.
    :param n_trials: Number of trials for hyperparameter tuning.
    :param time_column: Name of the time column in the DataFrame.
    :param dimension_column: Name of the dimension column in the DataFrame.
    :param spatial_x_column: Name of the spatial X column in the DataFrame.
    :param spatial_y_column: Name of the spatial Y column in the DataFrame.
    :param preprocessing: Flag to indicate if preprocessing should be applied.
    :param index: Flag to indicate if the time column is set as the index.
    :param is_multivariate: Flag to indicate if the data is multivariate.
    :param areaVStime: Parameter to determine how K-means clusters are produced.
    :return: Best parameters and results of the hyperparameter tuning.
    """

    # custom method that computes metrics for hyperparameter tuning for the 2 input cases
    def _compute_metrics_hypertuning(missing_df: pd.DataFrame,
                                     groundTruth_df: pd.DataFrame,
                                     filled_df: pd.DataFrame,
                                     time_column: str = '',
                                     dimension_column: str = '',
                                     spatial_x_column: str = '',
                                     spatial_y_column: str = '',
                                     preprocessing: bool = False,
                                     index: bool = False):
        if preprocessing:
            exclude_cols = [dimension_column, spatial_x_column, spatial_y_column]

            missing_df_selected = missing_df.loc[:,
                                  [col for col in missing_df.columns
                                   if col not in exclude_cols]]

            groundTruth_df_selected = groundTruth_df.loc[:,
                                      [col for col in groundTruth_df.columns
                                       if col not in exclude_cols]]

            filled_df_selected = filled_df.loc[:,
                                 [col for col in filled_df.columns
                                  if col not in exclude_cols]]

            metrics = compute_metrics(groundTruth_df_selected,
                                      missing_df_selected,
                                      filled_df_selected)
        else:
            if not index:
                missing_df_selected = missing_df.set_index(time_column)
                groundTruth_df_selected = groundTruth_df.set_index(time_column)
                filled_df_selected = filled_df.set_index(time_column)

                metrics = compute_metrics(groundTruth_df_selected,
                                          missing_df_selected,
                                          filled_df_selected)
            else:
                metrics = compute_metrics(groundTruth_df,
                                          missing_df,
                                          filled_df)

        return metrics

    if params is None:
        params = {}
    else:
        params = {key.lower(): value for key, value in params.items()}

    alg_results = {algorithm: {}}

    # Use missing dataframe as ground truth and introduce new missing dataframe
    if groundTruth_df is None:
        groundTruth_df = missing_df.copy()

        missing_df = run_gap_generation(ground_truth=groundTruth_df,
                                        time_column=time_column,
                                        dimension_column=dimension_column,
                                        spatial_x_column=spatial_x_column,
                                        spatial_y_column=spatial_y_column,
                                        train_params={},
                                        preprocessing=preprocessing,
                                        index=index)

    if algorithm.lower() == 'naive':
        # run imputation
        start_time = time.time()
        filled_df = run_imputation(missing=missing_df,
                                   algorithms=[algorithm],
                                   params={},
                                   time_column=time_column,
                                   dimension_column=dimension_column,
                                   spatial_x_column=spatial_x_column,
                                   spatial_y_column=spatial_y_column,
                                   areaVStime=areaVStime,
                                   is_multivariate=is_multivariate,
                                   preprocessing=preprocessing,
                                   index=index)[algorithm]
        exec_time = (time.time() - start_time)

        metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                               groundTruth_df=groundTruth_df,
                                               filled_df=filled_df,
                                               time_column=time_column,
                                               dimension_column=dimension_column,
                                               spatial_x_column=spatial_x_column,
                                               spatial_y_column=spatial_y_column,
                                               preprocessing=preprocessing,
                                               index=index)
        # metrics
        metrics['Execution Time'] = exec_time

        alg_results[algorithm]['metrics'] = metrics

        return {}, alg_results
    elif algorithm.lower() == 'cdmissingvaluerecovery':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            truncation = trial.suggest_categorical("truncation", model_parameters['truncation'])
            eps = trial.suggest_float("eps", model_parameters['eps'][0], model_parameters['eps'][1])
            parameters = {
                algorithm: {
                    "truncation": truncation,
                    "eps": eps
                }
            }
            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'softimpute':

        model_parameters = params[algorithm.lower()]
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

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'dynammo':
        model_parameters = params[algorithm.lower()]
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

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'iterativesvd':

        model_parameters = params[algorithm.lower()]
        already_seen = []

        def objective(trial):
            rank = trial.suggest_categorical("rank", model_parameters['rank'])

            parameters = {
                algorithm: {
                    "rank": rank
                }
            }

            if rank in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(rank)

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'rosl':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            rank = trial.suggest_categorical("rank", model_parameters['rank'])
            reg = trial.suggest_float("reg", model_parameters['reg'][0], model_parameters['reg'][1], log=True)
            parameters = {
                algorithm: {
                    "rank": rank,
                    "reg": reg
                }
            }
            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'ogdimpute':

        model_parameters = params[algorithm.lower()]
        already_seen = []

        def objective(trial):
            truncation = trial.suggest_categorical("truncation", model_parameters['truncation'])

            parameters = {
                algorithm: {
                    "truncation": truncation
                }
            }

            if truncation in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(truncation)

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'nmfmissingvaluerecovery':

        model_parameters = params[algorithm.lower()]
        already_seen = []

        def objective(trial):
            truncation = trial.suggest_categorical("truncation", model_parameters['truncation'])

            parameters = {
                algorithm: {
                    "truncation": truncation
                }
            }

            if truncation in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(truncation)

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'grouse':

        model_parameters = params[algorithm.lower()]
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

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'pca_mme':
        model_parameters = params[algorithm.lower()]
        already_seen = []

        def objective(trial):
            truncation = trial.suggest_categorical("truncation", model_parameters['truncation'])
            single_block = trial.suggest_categorical("single_block", model_parameters['single_block'])

            parameters = {
                algorithm: {
                    "truncation": truncation,
                    "single_block": single_block,
                }
            }
            value = str(truncation) + '_' + str(single_block)
            if value in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(value)

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'svt':

        model_parameters = params[algorithm.lower()]
        already_seen = []

        def objective(trial):
            tauScale = trial.suggest_categorical("tauScale", model_parameters['tauScale'])

            parameters = {
                algorithm: {
                    "tauScale": tauScale
                }
            }

            if tauScale in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(tauScale)

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'tkcm':

        model_parameters = params[algorithm.lower()]
        already_seen = []

        def objective(trial):
            truncation = trial.suggest_categorical("truncation", model_parameters['truncation'])

            parameters = {
                algorithm: {
                    "truncation": truncation
                }
            }

            if truncation in already_seen:
                raise optuna.exceptions.TrialPruned()
            else:
                already_seen.append(truncation)

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'spirit':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            k0 = trial.suggest_categorical("k0", model_parameters['k0'])
            w = trial.suggest_categorical("w", model_parameters['w'])
            lambda_par = trial.suggest_float("lambda", model_parameters['lambda'][0], model_parameters['lambda'][1])
            parameters = {
                algorithm: {
                    "k0": k0,
                    "w": w,
                    "lambda": lambda_par
                }
            }
            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'saits':
        model_parameters = params[algorithm.lower()]

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
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'

            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
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
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'timesnet':
        model_parameters = params[algorithm.lower()]

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
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
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
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'nonstationary_transformer':
        model_parameters = params[algorithm.lower()]

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
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
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
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']

            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'autoformer':
        model_parameters = params[algorithm.lower()]

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
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
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
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']
            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'brits':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            rnn_hidden_size = trial.suggest_categorical("rnn_hidden_size", model_parameters['rnn_hidden_size'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
            parameters = {
                algorithm: {
                    "rnn_hidden_size": rnn_hidden_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']
            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'csdi':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            n_layers = trial.suggest_categorical("n_layers", model_parameters['n_layers'])
            n_channels = trial.suggest_categorical("n_channels", model_parameters['n_channels'])
            n_heads = trial.suggest_categorical("n_heads", model_parameters['n_heads'])
            d_time_embedding = trial.suggest_categorical("d_time_embedding", model_parameters['d_time_embedding'])
            d_feature_embedding = trial.suggest_categorical("d_feature_embedding",
                                                            model_parameters['d_feature_embedding'])
            d_diffusion_embedding = trial.suggest_categorical("d_diffusion_embedding",
                                                              model_parameters['d_diffusion_embedding'])
            is_unconditional = trial.suggest_categorical("is_unconditional",
                                                         model_parameters['is_unconditional'])
            n_diffusion_steps = trial.suggest_categorical("n_diffusion_steps",
                                                          model_parameters['n_diffusion_steps'])
            beta_start = trial.suggest_categorical("beta_start", model_parameters['beta_start'])
            beta_end = trial.suggest_categorical("beta_end", model_parameters['beta_end'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
            parameters = {
                algorithm: {
                    "n_layers": n_layers,
                    "n_channels": n_channels,
                    "n_heads": n_heads,
                    "d_time_embedding": d_time_embedding,
                    "d_feature_embedding": d_feature_embedding,
                    "d_diffusion_embedding": d_diffusion_embedding,
                    "is_unconditional": is_unconditional,
                    "beta_start": beta_start,
                    "beta_end": beta_end,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "n_diffusion_steps": n_diffusion_steps,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']
            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'usgan':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            rnn_hidden_size = trial.suggest_categorical("rnn_hidden_size", model_parameters['rnn_hidden_size'])
            lambda_mse = trial.suggest_float("lambda_mse", model_parameters['lambda_mse'][0],
                                             model_parameters['lambda_mse'][1])
            hint_rate = trial.suggest_float("hint_rate", model_parameters['hint_rate'][0],
                                            model_parameters['hint_rate'][1])
            dropout = trial.suggest_categorical("dropout", model_parameters['dropout'])
            G_steps = trial.suggest_categorical("G_steps", model_parameters['G_steps'])
            D_steps = trial.suggest_categorical("D_steps", model_parameters['D_steps'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
            parameters = {
                algorithm: {
                    "rnn_hidden_size": rnn_hidden_size,
                    "lambda_mse": lambda_mse,
                    "hint_rate": hint_rate,
                    "dropout": dropout,
                    "G_steps": G_steps,
                    "D_steps": D_steps,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']
            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'mrnn':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            rnn_hidden_size = trial.suggest_categorical("rnn_hidden_size", model_parameters['rnn_hidden_size'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
            parameters = {
                algorithm: {
                    "rnn_hidden_size": rnn_hidden_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']
            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    elif algorithm.lower() == 'gpvae':
        model_parameters = params[algorithm.lower()]

        def objective(trial):
            latent_size = trial.suggest_categorical("latent_size", model_parameters['latent_size'])
            choices_tuple = tuple(tuple(choice) for choice in model_parameters['encoder_sizes'])
            encoder_sizes = trial.suggest_categorical("encoder_sizes", choices_tuple)
            choices_tuple2 = tuple(tuple(choice) for choice in model_parameters['decoder_sizes'])
            decoder_sizes = trial.suggest_categorical("decoder_sizes", choices_tuple2)
            kernel = trial.suggest_categorical("kernel", model_parameters['kernel'])
            beta = trial.suggest_categorical("beta", model_parameters['beta'])
            M = trial.suggest_categorical("M", model_parameters['M'])
            K = trial.suggest_categorical("K", model_parameters['K'])
            sigma = trial.suggest_categorical("sigma", model_parameters['sigma'])
            length_scale = trial.suggest_categorical("length_scale", model_parameters['length_scale'])
            kernel_scales = trial.suggest_categorical("kernel_scales", model_parameters['kernel_scales'])
            window_size = trial.suggest_categorical("window_size", model_parameters['window_size'])
            epochs = trial.suggest_categorical("epochs", model_parameters['epochs'])
            batch_size = trial.suggest_categorical("batch_size", model_parameters['batch_size'])
            num_workers = trial.suggest_categorical("num_workers", model_parameters['num_workers'])
            patience = trial.suggest_categorical("patience", model_parameters['patience'])
            if 'device' in model_parameters:
                device = model_parameters['device']
            else:
                device = None

            if not cuda.is_available():
                device = 'cpu'
            lr = trial.suggest_float("lr", model_parameters['lr'][0], model_parameters['lr'][1], log=True)
            weight_decay = trial.suggest_float("weight_decay", model_parameters['weight_decay'][0],
                                               model_parameters['weight_decay'][1])
            parameters = {
                algorithm: {
                    "latent_size": latent_size,
                    "encoder_sizes": encoder_sizes,
                    "decoder_sizes": decoder_sizes,
                    "kernel": kernel,
                    "beta": beta,
                    "M": M,
                    "K": K,
                    "sigma": sigma,
                    "length_scale": length_scale,
                    "kernel_scales": kernel_scales,
                    "window_size": window_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "device": device
                }
            }

            alg_results[algorithm][f'trial_number_{trial.number}'] = {}
            alg_results[algorithm][f'trial_number_{trial.number}']['params'] = parameters

            try:
                # run imputation
                start_time = time.time()
                filled_df = run_imputation(missing=missing_df,
                                           algorithms=[algorithm],
                                           params=parameters,
                                           time_column=time_column,
                                           dimension_column=dimension_column,
                                           spatial_x_column=spatial_x_column,
                                           spatial_y_column=spatial_y_column,
                                           areaVStime=areaVStime,
                                           is_multivariate=is_multivariate,
                                           preprocessing=preprocessing,
                                           index=index)[algorithm]
                exec_time = (time.time() - start_time)

                metrics = _compute_metrics_hypertuning(missing_df=missing_df,
                                                       groundTruth_df=groundTruth_df,
                                                       filled_df=filled_df,
                                                       time_column=time_column,
                                                       dimension_column=dimension_column,
                                                       spatial_x_column=spatial_x_column,
                                                       spatial_y_column=spatial_y_column,
                                                       preprocessing=preprocessing,
                                                       index=index)
                # metrics
                metrics['Execution Time'] = exec_time

                alg_results[algorithm][f'trial_number_{trial.number}']['metrics'] = metrics

                if np.isnan(metrics['Mean square error']):
                    raise optuna.exceptions.TrialPruned()
                return metrics['Mean square error']
            except:
                print('An exception occured, skip trial!')
                alg_results[algorithm][f'trial_number_{trial.number}']['Error'] = 'An exception occured, skip trial!'
                return np.inf
    if algorithm.lower() != 'naive':
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=2023))
        study.optimize(objective, n_trials=n_trials)
        alg_results[algorithm]['best_trial'] = {'number': study.best_trial.number}
        alg_results[algorithm]['best_trial'].update(alg_results[algorithm][f'trial_number_{study.best_trial.number}'])

    return alg_results[algorithm]['best_trial']['params'], alg_results
