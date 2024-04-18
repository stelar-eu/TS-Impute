import algorithms as alg
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import random
import math
from typing import Union
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pypots.imputation import SAITS, BRITS, CSDI, MRNN, USGAN, GPVAE, TimesNet
from sklearn.cluster import KMeans
import datetime as dt

__all__ = ['run_gap_generation', 'run_imputation', 'train_ensemble', 'run_imputation_ensemble',
           'fill_missing_values', 'fill_missing_values_ml', 'dataframe_preproc',
           'dataframe_reverse_preproc', 'z_normalize',
           'smooth_data', 'generate_custom_gaps',
           'fill_missing_with_lm', 'ensemble_model_run_imputation',
           'find_clusters']


def run_gap_generation(ground_truth: Union[str, pd.DataFrame], train_params: dict,
                       dimension_column: str = '', time_column: str = '', datetime_format: str = '',
                       spatial_x_column: str = '', spatial_y_column: str = '',
                       header: int = 0, sep: str = ',', preprocessing: bool = True, index: bool = False):
    """
    Introduce missing values to a dataframe using one of the available types.
    :param ground_truth: dataframe or a path to the .csv file which we introduce missing values
    :param train_params: dictionary with additional parameters for gap generation
    :param dimension_column: (optional) the name of the dimension column
    :param time_column: (optional) the name of the datetime column
    :param datetime_format: (optional) the format of the timestamp
    :param spatial_x_column: (optional) the name of the Spatial_X column
    :param spatial_y_column: (optional) the name of the Spatial_Y column
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :param sep: separator character to use for the csv
    :param preprocessing: if true it activates preprocessing function where DateTime in columns
    :param index: if true the datetime exists in the index of the dataframe. Don't change if you give a path.
    :return: dataframe with missing values
    """
    if preprocessing:
        df, spatial_x, spatial_y = dataframe_preproc(df_input=ground_truth, dimension_column=dimension_column,
                                                     time_column=time_column, datetime_format=datetime_format,
                                                     spatial_x_column=spatial_x_column,
                                                     spatial_y_column=spatial_y_column,
                                                     header=header, sep=sep)
    else:
        df = min_preproc(df_input=ground_truth, time_column=time_column,
                         datetime_format=datetime_format, header=header, sep=sep, index=index)

    gap_type = train_params['gap_type']
    miss_perc = train_params['miss_perc']
    gap_length = train_params['gap_length']
    max_gap_length = train_params['max_gap_length']
    max_gap_count = train_params['max_gap_count']

    data_gaps = generate_custom_gaps(data=df, gap_type=gap_type,
                                     miss_perc=miss_perc, gap_length=gap_length,
                                     max_gap_length=max_gap_length,
                                     max_gap_count=max_gap_count)
    if preprocessing:
        data_gaps = dataframe_reverse_preproc(df=data_gaps, dimension_column=dimension_column,
                                              spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                              spatial_x=spatial_x, spatial_y=spatial_y)
    else:
        data_gaps = reverse_min_preproc(df=data_gaps, time_column=time_column)

    return data_gaps


def run_imputation(missing: Union[str, pd.DataFrame], algorithms, params,
                   dimension_column: str = '', time_column: str = '', datetime_format: str = '',
                   spatial_x_column: str = '', spatial_y_column: str = '',
                   header: int = 0, sep: str = ',', is_multivariate: bool = False,
                   areaVStime: int = 0, preprocessing: bool = True, index: bool = False) -> (dict, pd.DataFrame):
    """
    Run imputation for each algorithm and return a list containing a dataframe with the results of each algorithm.
    :param missing: dataframe or a path to the .csv file with the missing values
    :param algorithms: list of algorithms to be used
    :param params: dictionary with the parameters for each algorithm
    :param dimension_column: (optional) the name of the dimension column
    :param time_column: (optional) the name of the datetime column
    :param datetime_format: (optional) the format of the timestamp
    :param spatial_x_column: (optional) the name of the Spatial_X column
    :param spatial_y_column: (optional) the name of the Spatial_Y column
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :param sep: separator character to use for the csv
    :param is_multivariate: if true we have a single multivariate timeseries and
    if false we have multiple timeseries with one feature.
    :param areaVStime: Determines how the K-means clusters are produced. If 0 we use the columns with names
    spatial_x_column and spatial_y_column (if they exist or else we set areaVStime to 1 if they do not) else we
    use the interpolated values in the columns that correspond to the timestamps.
    :param preprocessing: if true it activates preprocessing function where DateTime in columns
    :param index: if true the datetime exists in the index of the dataframe. Don't change if you give a path.
    :return: list of dataframes with the results
    """
    ml_algorithms = ['saits', 'brits', 'csdi', 'usgan', 'mrnn', 'gpvae', 'timesnet']

    if preprocessing:
        clusters_dict = find_clusters(df_input=missing, dimension_column=dimension_column,
                                      spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                      header=header, sep=sep, areaVStime=areaVStime)

        df, spatial_x, spatial_y = dataframe_preproc(df_input=missing, dimension_column=dimension_column,
                                                     time_column=time_column, datetime_format=datetime_format,
                                                     spatial_x_column=spatial_x_column,
                                                     spatial_y_column=spatial_y_column,
                                                     header=header, sep=sep)
    else:
        df = min_preproc(df_input=missing, time_column=time_column,
                         datetime_format=datetime_format, header=header, sep=sep, index=index)

        df_trans = dataframe_reverse_preproc(df=df, dimension_column='TimeSeries')

        clusters_dict = find_clusters(df_input=df_trans, dimension_column='TimeSeries',
                                      spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                      header=header, sep=sep, areaVStime=areaVStime)

    dict_imputed_dfs = dict()
    for alg in algorithms:
        new_alg_df = df.copy()
        for cluster in clusters_dict:
            df_cl = df[clusters_dict[cluster]]

            new_cluster_alg_df = pd.DataFrame()
            if alg.lower() in ml_algorithms:
                new_cluster_alg_df = fill_missing_values_ml(df_cl, alg.lower(), params[str(alg)], is_multivariate)
            else:
                new_cluster_alg_df = fill_missing_values(df_cl, alg, params[str(alg)])

            new_alg_df[clusters_dict[cluster]] = new_cluster_alg_df.values

        if preprocessing:
            new_alg_df = dataframe_reverse_preproc(df=new_alg_df, dimension_column=dimension_column,
                                                   spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                                   spatial_x=spatial_x, spatial_y=spatial_y)
        else:
            new_alg_df = reverse_min_preproc(df=new_alg_df, time_column=time_column)

        dict_imputed_dfs[str(alg)] = new_alg_df

    return dict_imputed_dfs


def train_ensemble(ground_truth: Union[str, pd.DataFrame], algorithms, params,
                   train_params, dimension_column: str = '', time_column: str = '',
                   datetime_format: str = '', spatial_x_column: str = '',
                   spatial_y_column: str = '', header: int = 0, sep: str = ',',
                   is_multivariate: bool = False, areaVStime: int = 0,
                   preprocessing: bool = True, index: bool = False) -> (
        XGBRegressor, dict):
    """
    Take the ground truth dataframe and a second dataframe with gaps. On this dataframe run imputation for each
    algorithm and use the results of the algorithms to train a model. Then, run again imputation for each
    algorithm and use the model to make a prediction using as input the imputation results of each algorithm.
    Finally, extract metrics by using the real values of the ground truth dataframe against the imputed values
    resulting from the above-mentioned algorithms and trained model. Return the trained model and a dictionary
    each of the algorithms and the trained model as keys and a dict with various metric-value as values.
    :param ground_truth: dataframe or a path to the .csv file used as ground truth to train the linear model
    :param algorithms: list of algorithms to be used
    :param params: dictionary with the parameters for each algorithm
    :param train_params: dictionary with additional parameters for the ensemble model
                         used for normalization, smoothing and gap generation
    :param dimension_column: (optional) the name of the dimension column
    :param time_column: (optional) the name of the datetime column
    :param datetime_format: (optional) the format of the timestamp
    :param spatial_x_column: (optional) the name of the Spatial_X column
    :param spatial_y_column: (optional) the name of the Spatial_Y column
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :param sep: separator character to use for the csv
    :param is_multivariate: if true we have a single multivariate timeseries and
    if false we have multiple timeseries with one feature.
    :param areaVStime: Determines how the K-means clusters are produced. If 0 we use the columns with names
    spatial_x_column and spatial_y_column (if they exist or else we set areaVStime to 1 if they do not) else we
    use the interpolated values in the columns that correspond to the timestamps.
    :param preprocessing: if true it activates preprocessing function where DateTime in columns
    :param index: if true the datetime exists in the index of the dataframe. Don't change if you give a path.
    :return: resulting model, dictionary of metrics
    """

    if preprocessing:
        clusters_dict = find_clusters(df_input=ground_truth, dimension_column=dimension_column,
                                      spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                      header=header, sep=sep, areaVStime=areaVStime)

        df, spatial_x, spatial_y = dataframe_preproc(df_input=ground_truth, dimension_column=dimension_column,
                                                     time_column=time_column, datetime_format=datetime_format,
                                                     spatial_x_column=spatial_x_column,
                                                     spatial_y_column=spatial_y_column,
                                                     header=header, sep=sep)
    else:
        df = min_preproc(df_input=ground_truth, time_column=time_column,
                         datetime_format=datetime_format, header=header, sep=sep, index=index)

        df_trans = dataframe_reverse_preproc(df=df, dimension_column='TimeSeries')

        clusters_dict = find_clusters(df_input=df_trans, dimension_column='TimeSeries',
                                      spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                      header=header, sep=sep, areaVStime=areaVStime)

    smooth = train_params['smooth']
    window = train_params['window']
    order = train_params['order']
    normalize = train_params['normalize']
    gap_type = train_params['gap_type']
    miss_perc = train_params['miss_perc']
    gap_length = train_params['gap_length']
    max_gap_length = train_params['max_gap_length']
    max_gap_count = train_params['max_gap_count']

    if normalize:
        df = z_normalize(df)

    # Smooth the data
    if smooth:
        df = smooth_data(df, window, order)

    # Generate custom gaps
    data_gaps = generate_custom_gaps(data=df, gap_type=gap_type,
                                     miss_perc=miss_perc, gap_length=gap_length,
                                     max_gap_length=max_gap_length,
                                     max_gap_count=max_gap_count)
    # clustering

    # Train Ensemble Model
    training = data_gaps.copy()
    # inside ensemble_model_run_imputation for each algorithm for each cluster
    # take only the columns in dict[cluster] then create filled training dataframe with isin
    # flatten and add to resulting dataframe(to keep the original flow)
    pred_ms_ts_original = ensemble_model_run_imputation(training, algorithms,
                                                        params, clusters_dict, is_multivariate)

    real_np = df.to_numpy().flatten()
    new_np = training.to_numpy().flatten()
    real_val = real_np[np.isnan(new_np)]
    y = pd.Series(real_val)
    xgbr_model = XGBRegressor(learning_rate=0.1, max_depth=2, n_estimators=200, random_state=1, subsample=0.75)
    xgbr_model.fit(pred_ms_ts_original, y)

    # Testing the results of the algorithms and the ensemble model
    testing = data_gaps
    pred_ms_ts = ensemble_model_run_imputation(testing, algorithms, params, clusters_dict, is_multivariate)
    pred_ms_ts['Ensemble_Model'] = xgbr_model.predict(pred_ms_ts)

    pred_ms_ts['True'] = real_val

    # Calculate metrics
    metrics = dict()
    for col in pred_ms_ts.columns[pred_ms_ts.columns != 'True']:
        metrics[col] = dict()

        mae = mean_absolute_error(pred_ms_ts["True"], pred_ms_ts[col])
        mse = mean_squared_error(pred_ms_ts["True"], pred_ms_ts[col])
        rmse = mean_squared_error(pred_ms_ts["True"], pred_ms_ts[col], squared=False)
        r2 = r2_score(pred_ms_ts["True"], pred_ms_ts[col])

        euclidean_distance = np.sqrt(np.sum(
            [(pred_ms_ts["True"] - pred_ms_ts[col]) * (pred_ms_ts["True"] - pred_ms_ts[col]) for a, b in
             zip(pred_ms_ts["True"], pred_ms_ts[col])]))

        metrics[col]['mae'] = mae
        metrics[col]['mse'] = mse
        metrics[col]['rmse'] = rmse
        metrics[col]['r2'] = r2
        metrics[col]['euclidean_distance'] = euclidean_distance

    return xgbr_model, metrics


def run_imputation_ensemble(missing: Union[str, pd.DataFrame], algorithms: dict,
                            params: dict, model: XGBRegressor, dimension_column: str = '',
                            time_column: str = '', datetime_format: str = '',
                            spatial_x_column: str = '', spatial_y_column: str = '',
                            header: int = 0, sep: str = ',',
                            is_multivariate: bool = False, areaVStime: int = 0,
                            preprocessing: bool = True,
                            index: bool = False) -> pd.DataFrame:
    """
    Run imputation and create a dataframe with columns the filled missing values of each algorithm,
    then use the model to make a prediction using as input the imputation results of each algorithm
    and return a dataframe with the imputed values resulting by the model predictions.
    :param missing: dataframe or a path to the .csv file with the missing values
    :param algorithms: list of algorithms to be used
    :param params: dictionary with the parameters for each algorithm
    :param model: the ensemble model
    :param dimension_column: the name of the dimension column
    :param time_column: the name of the datetime column
    :param datetime_format: the format of the timestamp
    :param spatial_x_column: (optional) the name of the Spatial_X column
    :param spatial_y_column: (optional) the name of the Spatial_Y column
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :param sep: separator character to use for the csv
    :param is_multivariate: if true we have a single multivariate timeseries and
    if false we have multiple timeseries with one feature.
    :param areaVStime: Determines how the K-means clusters are produced. If 0 we use the columns with names
    spatial_x_column and spatial_y_column (if they exist or else we set areaVStime to 1 if they do not) else we
    use the interpolated values in the columns that correspond to the timestamps.
    :param preprocessing: if true it activates preprocessing function where DateTime in columns
    :param index: if true the datetime exists in the index of the dataframe. Don't change if you give a path.
    :return: dataframe with the results
    """

    if preprocessing:
        clusters_dict = find_clusters(df_input=missing, dimension_column=dimension_column,
                                      spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                      header=header, sep=sep, areaVStime=areaVStime)

        df, spatial_x, spatial_y = dataframe_preproc(df_input=missing, dimension_column=dimension_column,
                                                     time_column=time_column, datetime_format=datetime_format,
                                                     spatial_x_column=spatial_x_column,
                                                     spatial_y_column=spatial_y_column,
                                                     header=header, sep=sep)
    else:
        df = min_preproc(df_input=missing, time_column=time_column,
                         datetime_format=datetime_format, header=header, sep=sep, index=index)

        df_trans = dataframe_reverse_preproc(df=df, dimension_column='TimeSeries')

        clusters_dict = find_clusters(df_input=df_trans, dimension_column='TimeSeries',
                                      spatial_x_column=spatial_x_column, spatial_y_column=spatial_y_column,
                                      header=header, sep=sep, areaVStime=areaVStime)

    pred_ms_ts = ensemble_model_run_imputation(df, algorithms, params, clusters_dict, is_multivariate)
    pred_model_ms_ts = model.predict(pred_ms_ts)

    model_pred_df = fill_missing_with_lm(df, pred_model_ms_ts)
    if preprocessing:
        model_pred_df = dataframe_reverse_preproc(model_pred_df, dimension_column,
                                                  spatial_x, spatial_y,
                                                  spatial_x_column, spatial_y_column)
    else:
        model_pred_df = reverse_min_preproc(df=model_pred_df, time_column=time_column)

    return model_pred_df


def fill_missing_values(time_series: pd.DataFrame, algorithm, var_dict):
    """
    This is a missing value imputation method that uses a specific algorithm and its parameters to fill the missing
    values of a pandas Dataframe that has time series as columns.
    The available algorithms are divided in the following categories:
    1. Matrix Based
        1. SVD - Based
            1. IterativeSVD (SVDImpute)
            2. SoftImpute
            3. SVT
        2. Centroid Decomposition - Based
            1. CDMissingValueRecovery (CDRec)
        3. PCA - Based
            1. GROUSE
            2. SPIRIT
            3. ROSL
            4. PCA_MME
        4. Matrix Factorization - Based
            1. NMFMissingValueRecovery (TeNMF)
    2. Pattern Based
        1. OGDImpute
        2. TKCM
        3. DynaMMo
    3. Other
        1. MeanImpute
        2. LinearImpute
        3. ZaeroImpute

    :param time_series: A pandas DataFrame with missing values containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param algorithm: The name of the algorithm to be used for the imputation.
    :type algorithm: string
    :param var_dict: The parameters of the algorithm in a dictionary.
    :type var_dict: dict
    :return:
        -c_timeseries (pandas.DataFrame) - A pandas DataFrame with no missing values containing the loaded time series
        as columns.
    """
    c_timeseries = time_series.copy()
    lists = c_timeseries.values.tolist()
    m_ndarray = np.array(lists)
    algorithm = algorithm.lower()
    if algorithm == 'ogdimpute':
        svd: np.ndarray = alg.OGDImpute.doOGDImpute(m_ndarray, var_dict['truncation'])
    elif algorithm == 'rosl':
        svd: np.ndarray = alg.ROSL.doROSL(m_ndarray, var_dict['rank'], var_dict['reg'])
    elif algorithm == 'cdmissingvaluerecovery':
        svd: np.ndarray = alg.CDMissingValueRecovery.doCDMissingValueRecovery(m_ndarray, var_dict['truncation'],
                                                                              var_dict['eps'])
    elif algorithm == 'grouse':
        m_ndarray = np.transpose(m_ndarray)
        svd: np.ndarray = alg.GROUSE.doGROUSE(m_ndarray, var_dict['max_rank'])
    elif algorithm == 'dynammo':
        m_ndarray = np.transpose(m_ndarray)
        svd: np.ndarray = alg.DynaMMo.doDynaMMo(m_ndarray, var_dict['H'], var_dict['max_iter'], var_dict['fast'])
    elif algorithm == 'iterativesvd':
        svd: np.ndarray = alg.IterativeSVD.doIterativeSVD(m_ndarray, var_dict['rank'])
    elif algorithm == 'softimpute':
        svd: np.ndarray = alg.SoftImpute.doSoftImpute(m_ndarray, var_dict['max_rank'])
    elif algorithm == 'pca_mme':
        m_ndarray = np.transpose(m_ndarray)
        svd: np.ndarray = alg.PCA_MME.doPCA_MME(m_ndarray, var_dict['truncation'], var_dict['single_block'])
    elif algorithm == 'linearimpute':
        svd: np.ndarray = alg.LinearImpute.doLinearImpute(m_ndarray)

    elif algorithm == 'meanimpute':
        svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)
    elif algorithm == 'nmfmissingvaluerecovery':
        svd: np.ndarray = alg.NMFMissingValueRecovery.doNMFRecovery(m_ndarray, var_dict['truncation'])
    elif algorithm == 'svt':
        svd: np.ndarray = alg.SVT.doSVT(m_ndarray, var_dict['tauScale'])
    elif algorithm == 'zeroimpute':
        svd: np.ndarray = alg.ZeroImpute.doZeroImpute(m_ndarray)
    elif algorithm == 'spirit':
        svd: np.ndarray = alg.Spirit.doSpirit(m_ndarray, var_dict['k0'], var_dict['w'], var_dict['lambda'])
    elif algorithm == 'tkcm':
        svd: np.ndarray = alg.TKCM.doTKCM(m_ndarray, var_dict['truncation'])
    else:
        svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)

    if algorithm == 'grouse' or algorithm == 'dynammo' or algorithm == 'pca_mme':
        svd = np.transpose(svd)
        m_ndarray = np.transpose(m_ndarray)
    c_timeseries[c_timeseries.isnull()] = svd
    if c_timeseries.isnull().values.any():
        svd: np.ndarray = alg.MeanImpute.doMeanImpute(m_ndarray)
        c_timeseries[c_timeseries.isnull()] = svd

    return c_timeseries


def fill_missing_values_ml(time_series: pd.DataFrame, algorithm, var_dict, is_multivariate: bool = False):
    """
    This is a missing value imputation method that uses a specific PYPOTS algorithm
    and its parameters to fill the missing values of a pandas Dataframe that has
    time series as columns.
    The available algorithms are divided in the following categories:
    SAITS
    BRITS
    CSDI
    USGAN
    MRNN
    GPVAE
    TIMESNET

    :param time_series: A pandas DataFrame with missing values containing the loaded time series as columns.
    :type time_series: pandas.DataFrame
    :param algorithm: The name of the algorithm to be used for the imputation.
    :type algorithm: string
    :param var_dict: The parameters of the algorithm in a dictionary.
    :type var_dict: dict
    :param is_multivariate: if true we have a single multivariate timeseries and
    if false we have multiple timeseries with one feature.
    :type is_multivariate: bool
    :return:
        -imputed_time_series (pandas.DataFrame) - A pandas DataFrame with no missing values containing the loaded time series
        as columns.
    """
    model = None
    num_timestamps = time_series.shape[0]
    num_columns = time_series.shape[1]
    if not is_multivariate:
        num_samples = num_columns
        num_features = 1
        dataset = {"X": time_series.values.T.reshape(num_samples, num_timestamps, -1)}
    else:
        num_samples = 1
        num_features = num_columns
        dataset = {"X": np.array([time_series.values])}
    algorithm = algorithm.lower()
    if algorithm == 'saits':
        n_layers = var_dict['n_layers']
        d_model = var_dict['d_model']
        d_inner = var_dict['d_inner']
        n_heads = var_dict['n_heads']
        d_k = var_dict['d_k']
        d_v = var_dict['d_v']
        dropout = var_dict['dropout']
        epochs = var_dict['epochs']
        batch_size = var_dict['batch_size']
        ORT_weight = var_dict['ORT_weight']
        MIT_weight = var_dict['MIT_weight']
        attn_dropout = var_dict['attn_dropout']
        diagonal_attention_mask = var_dict['diagonal_attention_mask']
        num_workers = var_dict['num_workers']

        model = SAITS(n_steps=num_timestamps, n_features=num_features, n_layers=n_layers,
                      d_model=d_model, d_inner=d_inner, n_heads=n_heads,
                      d_k=d_k, d_v=d_v, dropout=dropout, epochs=epochs, batch_size=batch_size,
                      ORT_weight=ORT_weight, MIT_weight=MIT_weight,
                      attn_dropout=attn_dropout,
                      diagonal_attention_mask=diagonal_attention_mask,
                      num_workers=num_workers)

    elif algorithm == 'brits':
        rnn_hidden_size = var_dict['rnn_hidden_size']
        epochs = var_dict['epochs']
        batch_size = var_dict['batch_size']
        num_workers = var_dict['num_workers']

        model = BRITS(n_steps=num_timestamps, n_features=num_features,
                      rnn_hidden_size=rnn_hidden_size, epochs=epochs,
                      batch_size=batch_size, num_workers=num_workers)

    elif algorithm == 'csdi':
        n_layers = var_dict['n_layers']
        n_channels = var_dict['n_channels']
        n_heads = var_dict['n_heads']
        d_time_embedding = var_dict['d_time_embedding']
        d_feature_embedding = var_dict['d_feature_embedding']
        d_diffusion_embedding = var_dict['d_diffusion_embedding']
        is_unconditional = var_dict['is_unconditional']
        beta_start = var_dict['beta_start']
        beta_end = var_dict['beta_end']
        n_diffusion_steps = var_dict['n_diffusion_steps']
        epochs = var_dict['epochs']
        batch_size = var_dict['batch_size']
        num_workers = var_dict['num_workers']

        model = CSDI(n_features=num_features, n_layers=n_layers,
                     n_channels=n_channels, n_heads=n_heads,
                     d_time_embedding=d_time_embedding,
                     d_feature_embedding=d_feature_embedding,
                     d_diffusion_embedding=d_diffusion_embedding,
                     is_unconditional=is_unconditional,
                     target_strategy='random',
                     beta_start=beta_start, beta_end=beta_end, epochs=epochs,
                     batch_size=batch_size, n_diffusion_steps=n_diffusion_steps,
                     num_workers=num_workers)

    elif algorithm == 'usgan':
        rnn_hidden_size = var_dict['rnn_hidden_size']
        lambda_mse = var_dict['lambda_mse']
        hint_rate = var_dict['hint_rate']
        dropout_rate = var_dict['dropout']
        G_steps = var_dict['G_steps']
        D_steps = var_dict['D_steps']
        epochs = var_dict['epochs']
        batch_size = var_dict['batch_size']
        num_workers = var_dict['num_workers']

        model = USGAN(n_steps=num_timestamps, n_features=num_features,
                      rnn_hidden_size=rnn_hidden_size, lambda_mse=lambda_mse,
                      hint_rate=hint_rate, dropout=dropout,
                      G_steps=G_steps, D_steps=D_steps, epochs=epochs,
                      batch_size=batch_size, num_workers=num_workers)

    elif algorithm == 'mrnn':
        rnn_hidden_size = var_dict['rnn_hidden_size']
        epochs = var_dict['epochs']
        batch_size = var_dict['batch_size']
        num_workers = var_dict['num_workers']

        model = MRNN(n_steps=num_timestamps, n_features=num_features,
                     rnn_hidden_size=rnn_hidden_size, epochs=epochs,
                     batch_size=batch_size, num_workers=num_workers)

    elif algorithm == 'gpvae':
        latent_size = var_dict['latent_size']
        encoder_sizes = tuple(var_dict['encoder_sizes'])
        decoder_sizes = tuple(var_dict['decoder_sizes'])
        kernel = var_dict['kernel']
        beta = var_dict['beta']
        M = var_dict['M']
        K = var_dict['K']
        sigma = var_dict['sigma']
        length_scale = var_dict['length_scale']
        kernel_scales = var_dict['kernel_scales']
        window_size = var_dict['window_size']
        epochs = var_dict['epochs']
        batch_size = var_dict['batch_size']
        num_workers = var_dict['num_workers']

        model = GPVAE(n_steps=num_timestamps, n_features=num_features,
                      latent_size=latent_size, encoder_sizes=encoder_sizes,
                      decoder_sizes=decoder_sizes,
                      kernel=kernel, beta=beta,
                      M=M, K=K, sigma=sigma,
                      length_scale=length_scale, kernel_scales=kernel_scales,
                      window_size=window_size, epochs=epochs,
                      batch_size=batch_size, num_workers=num_workers)

    elif algorithm == 'timesnet':
        n_layers = var_dict['n_layers']
        top_k = var_dict['top_k']
        d_model = var_dict['d_model']
        d_ffn = var_dict['d_ffn']
        n_kernels = var_dict['n_kernels']
        dropout = var_dict['dropout']
        epochs = var_dict['epochs']
        apply_nonstationary_norm = var_dict['apply_nonstationary_norm']
        batch_size = var_dict['batch_size']
        num_workers = var_dict['num_workers']

        model = TimesNet(n_steps=num_timestamps, n_features=num_features,
                         n_layers=n_layers, top_k=top_k, d_model=d_model,
                         d_ffn=d_ffn, n_kernels=n_kernels, dropout=dropout,
                         epochs=epochs, apply_nonstationary_norm=apply_nonstationary_norm,
                         batch_size=batch_size, num_workers=num_workers)

    model.fit(train_set=dataset)
    imputation = model.predict(dataset)
    imputed_time_series = time_series.copy()
    if not is_multivariate:
        imputed_time_series[:] = imputation['imputation'].T.reshape(num_timestamps, num_columns)
    else:
        imputed_time_series[:] = imputation['imputation'].reshape(num_timestamps, num_columns)
    return imputed_time_series


# -------- Other functions --------
def dataframe_preproc(df_input: Union[str, pd.DataFrame], dimension_column: str,
                      time_column: str = '', datetime_format: str = '',
                      spatial_x_column: str = '', spatial_y_column: str = '',
                      header: int = 0, sep: str = ',') -> [pd.DataFrame, Union[None, pd.Series],
                                                           Union[None, pd.Series]]:
    """
    Preprocess the dataframe to be used in the imputation
    :param df_input: dataframe or a path to the .csv file that needs preprocessing
    :param dimension_column: the name of the dimension column
    :param time_column: (optional) the name of the datetime column that will be inserted
    :param datetime_format: (optional) the format of the timestamp
    :param spatial_x_column: (optional) the name of the Spatial_X column
    :param spatial_y_column: (optional) the name of the Spatial_Y column
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :param sep: separator character to use for the csv
    :return: processed dataframe
    """

    if isinstance(df_input, str):
        df = pd.read_csv(df_input, header=header, sep=sep)
    else:
        df = df_input

    columns = list(df.columns)
    spatial_x = None
    if spatial_x_column in columns:
        spatial_x = df[spatial_x_column].copy()
        df = df.drop(spatial_x_column, axis=1)

    spatial_y = None
    if spatial_y_column in columns:
        spatial_y = df[spatial_y_column].copy()
        df = df.drop(spatial_y_column, axis=1)

    df = df.set_index(dimension_column)
    df = df.transpose()
    df.columns.name = ''
    df.reset_index(inplace=True, names=['index'])

    if time_column == '':
        time_column = 'Time'
    df = df.rename(columns={'index': time_column})

    if datetime_format != '':
        df[time_column] = pd.to_datetime(df[time_column], format=datetime_format)
    df = df.set_index(time_column)

    return df, spatial_x, spatial_y


def dataframe_reverse_preproc(df: pd.DataFrame, dimension_column: str,
                              spatial_x: Union[None, pd.Series] = None,
                              spatial_y: Union[None, pd.Series] = None,
                              spatial_x_column: str = '',
                              spatial_y_column: str = '') -> pd.DataFrame:
    """
    Reverse the preprocessing of the dataframe resulting from each imputation
    :param df: dataframe or a path to the .csv file that needs preprocessing
    :param dimension_column: the name of the dimension column
    :param spatial_x: (optional) the values of the Spatial_X column
    :param spatial_y: (optional) the values of the Spatial_Y column
    :param spatial_x_column: (optional) the name of the Spatial_X column
    :param spatial_y_column: (optional) the name of the Spatial_Y column
    :return: dataframe resulting from each imputation with the same style as the input dataframe
    """
    df_t: pd.DataFrame = df.transpose()
    if spatial_y is not None:
        df_t.insert(0, spatial_y_column, spatial_y.values)

    if spatial_x is not None:
        df_t.insert(0, spatial_x_column, spatial_x.values)

    df_t.reset_index(inplace=True, names=[dimension_column])
    df_t.columns.name = ''

    return df_t


def z_normalize(data):
    """
    Normalize the data.
    :param data: dataframe with the data
    :return: dataframe with the data normalized
    """
    return (data - data.mean()) / data.std()


def smooth_data(data, window, order):
    """
    Smooth the data.
    :param data: dataframe with the data
    :param window: the length of the filter window
    :param order: the order of the polynomial used to fit the samples
    :return: dataframe with the data smoothed
    """
    return data.apply(lambda x: savgol_filter(x, window, order))


def generate_custom_gaps(data: pd.DataFrame,
                         gap_type: str = 'random',
                         miss_perc: float = 0.1,
                         gap_length: int = 10,
                         max_gap_length: int = 0,
                         max_gap_count: int = 0):
    """
    Generate custom gaps for the data.
    :param data: dataframe with the data
    :param gap_type: type of gap to be generated
    :param miss_perc: float between 0 and 1 which determines the size of the gap.
    :param gap_length: int between 0 and 100 that determines the size of the gap
    :param max_gap_length: maximum length of the gap
    :param max_gap_count: maximum number of gaps
    :return: dataframe with the data with gaps

    Args:
        gap_length:
    """

    data_gaps = data.copy()
    random.seed(18931)

    if gap_type == 'random':
        """
        Needs max_gap_length and max_gap_count.
        """
        for col in data_gaps.columns:
            num_missing = random.randint(1, max_gap_count)
            for i in range(num_missing):
                start = random.randint(0, len(data_gaps) - max_gap_length)
                end = start + random.randint(1, max_gap_length)
                if start == 0:
                    start = 1
                if end == len(data_gaps):
                    end = len(data_gaps) - 1
                data_gaps[col][start:end] = np.nan

    if gap_type == 'single':
        """
        This method introduces missing values, on a pandas DataFrame, on the 1st time series column where the position
        of the first missing value is at 5% of 1st series from the top and the size of a single block varies based on the
        given miss_perc value.
        
        Needs miss_perc.
        """

        # N = length of time series, M = number of time series
        N, M = data_gaps.shape

        # choose starting pos at 5% of the above chosen time series from the top
        top_five_perc = (5 * N) // 100

        # name of the chosen time series in the dataframe
        col = data_gaps.columns[0]

        # maximum NaN pos in the chosen time series with starting point
        # the top_five_perc value to cover miss_perc of the series
        max_pos = int(miss_perc * N + top_five_perc)

        data_gaps.iloc[top_five_perc:max_pos].loc[:, col] = np.nan

    if gap_type == 'no_overlap':
        """
        This method introduces missing values, on a pandas DataFrame, on all the columns (time series).
        The size of each missing block is length of time series/number of time series and the position of the first missing
        value in each time series is at column_index*(length of time series/number of time series).
        """

        # N = length of time series, M = number of time series
        N, M = data_gaps.shape

        miss_size = N // M
        incr = 0
        for i in range(len(data_gaps.columns)):
            col = data_gaps.columns[i]
            data_gaps.iloc[incr:((i + 1) * miss_size)].loc[:, col] = np.nan
            incr = ((i + 1) * miss_size)

    if gap_type == 'blackout':
        """
        This method introduces missing values, on a pandas DataFrame, on all the columns (time series).
        We introduce missing blocks of size = m_rows on all the columns (time series) where the position
        of the first missing value is at 5% of all series from the top.
        
        Needs gap_length.
        """

        # N = length of time series, M = number of time series
        N, M = data_gaps.shape

        # choose starting pos at 5% of the time series from the top
        top_five_perc = (5 * N) // 100

        # maximum NaN pos in the time series with starting point the top_five_perc value to cover m_rows of the series
        max_pos = int(gap_length + top_five_perc)

        data_gaps.iloc[top_five_perc:max_pos - 1].loc[:, data_gaps.columns] = np.nan

    return data_gaps


def fill_missing_with_lm(df: pd.DataFrame, lm_pred):
    """
    Fill missing values with the linear regression prediction.
    :param df: dataframe with the missing values
    :param lm_pred: linear regression prediction
    :return: dataframe with the missing values filled
    """
    new_df = df.copy()

    def is_nan(value):
        return math.isnan(float(value))

    i = 0
    for index, row in new_df.iterrows():
        for j in range(len(row)):
            if is_nan(row.iloc[j]):
                row.iloc[j] = lm_pred[i]
                i = i + 1
        new_df.loc[index] = row
    return new_df


def ensemble_model_run_imputation(missing: pd.DataFrame, algorithms,
                                  params, clusters_dict: dict = None,
                                  is_multivariate: bool = False) -> pd.DataFrame:
    """
    Run imputation for each algorithm and return a dataframe with the results.
    :param missing: dataframe with the missing values or a path to the .csv file
    :param algorithms: list of algorithms to be used
    :param params: dictionary with the parameters for each algorithm
    :param clusters_dict: dictionary with a cluster as key and a list with timeseries ids as value
    :param is_multivariate: if true we have a single multivariate timeseries and
    if false we have multiple timeseries with one feature.
    :return: dataframe with the results
    """

    if clusters_dict is None:
        clusters_dict = dict()

    ml_algorithms = ['saits', 'brits', 'csdi', 'usgan', 'mrnn', 'gpvae', 'timesnet']
    pred_ms_ts = pd.DataFrame()
    new_np = missing.to_numpy().flatten()

    if not clusters_dict:
        clusters_dict = {0: missing.columns.values.tolist()}

    for alg in algorithms:
        new_alg_df = missing.copy()
        for cluster in clusters_dict:
            df_cl = missing[clusters_dict[cluster]]

            new_cluster_alg_df = pd.DataFrame()
            if alg.lower() in ml_algorithms:
                new_cluster_alg_df = fill_missing_values_ml(df_cl, alg.lower(), params[str(alg)], is_multivariate)
            else:
                new_cluster_alg_df = fill_missing_values(df_cl, alg, params[str(alg)])

            new_alg_df[clusters_dict[cluster]] = new_cluster_alg_df
        ts_np = new_alg_df.to_numpy().flatten()
        pred_ms_ts[str(alg)] = pd.Series(ts_np[np.isnan(new_np)])

    return pred_ms_ts


def find_clusters(df_input: Union[str, pd.DataFrame], dimension_column: str, spatial_x_column: str = '',
                  spatial_y_column: str = '',
                  header: int = 0, sep: str = ',', areaVStime: int = 0) -> [dict]:
    """
    :param df_input: dataframe or a path to the .csv file that needs preprocessing
    :param dimension_column: the name of the dimension column
    :param spatial_x_column: (optional) the name of the Spatial_X column
    :param spatial_y_column: (optional) the name of the Spatial_Y column
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :param sep: separator character to use for the csv
    :param areaVStime: Determines how the K-means clusters are produced. If 0 we use the columns with names spatial_x_column and spatial_y_column (if they exist or else we set areaVStime to 1 if they do not) else we use the interpolated values in the columns that correspond to the timestamps.
    :return: dictionary with a cluster as key and a list with timeseries ids as value
    """
    if isinstance(df_input, str):
        df = pd.read_csv(df_input, header=header, sep=sep)
    else:
        df = df_input

    columns = df.columns.values.tolist()
    columns.remove(dimension_column)
    # if spatial column exist remove them from columns list
    spatial_columns = 0
    if spatial_x_column in columns:
        columns.remove(spatial_x_column)
        spatial_columns += 1

    if spatial_y_column in columns:
        spatial_columns += 1
        columns.remove(spatial_y_column)

    if spatial_columns != 2:
        areaVStime = 1

    if areaVStime == 0:
        no_clusters = max(1, math.floor(len(df) / 100))
        kmeans = KMeans(n_clusters=no_clusters, random_state=0, n_init='auto').fit(
            df[[spatial_x_column, spatial_y_column]])
        df['cluster'] = kmeans.labels_
    else:
        no_clusters = max(1, math.floor(len(df) / 100))
        df_interp = df[columns].interpolate(method='linear', axis=1, limit_direction='both')
        kmeans = KMeans(n_clusters=no_clusters, random_state=0, n_init='auto').fit(df_interp)
        df['cluster'] = kmeans.labels_

    cluster_dict = dict(zip(df['cluster'].unique(), df.groupby('cluster')[dimension_column].apply(list)))
    df.drop(columns=['cluster'], inplace=True)
    return cluster_dict


def min_preproc(df_input: Union[str, pd.DataFrame],
                time_column: str = '', datetime_format: str = '',
                header: int = 0, sep: str = ',', index: bool = False):
    """
    Preprocess the dataframe to be used in the imputation
    :param df_input: dataframe or a path to the .csv file
    :param time_column: the name of the datetime column
    :param datetime_format: (optional) the format of the timestamp
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :param sep: separator character to use for the csv
    :param index: if true the datetime exists in the index of the dataframe
    :return: processed dataframe
    """

    if isinstance(df_input, str):
        df = pd.read_csv(df_input, header=header, sep=sep)
    else:
        df = df_input

    if index:
        if datetime_format != '':
            df.index = pd.to_datetime(df.index, format=datetime_format)
    else:
        if datetime_format != '':
            df[time_column] = pd.to_datetime(df[time_column], format=datetime_format)
        df = df.set_index(time_column)

    return df


def reverse_min_preproc(df: pd.DataFrame,
                        time_column: str = ''):
    """
    Preprocess the dataframe to be used in the imputation
    :param df: the resulting dataframe
    :param time_column: the name of the datetime column
    :return: processed dataframe
    """

    df.reset_index(inplace=True, names=[time_column])
    df.columns.name = ''

    return df
