import pandas as pd
import time
import numpy as np
from stelarImputation import imputation as tsi
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
        start2 = start + int((1 - perc)*length)
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


def test_imputation_metrics(input_folder: str, input_files: list, input_file: str, algorithm: str, params: dict, perc: float, length: int, seed: int):

    time_column='Time'
    preprocessing=False
    index=True
    is_multivariate = True
    data_format = '%d/%m/%Y %H:%M'
    sheet_name='Worksheet'
    index_col=0
    skiprows=4
    skipfooter=3

    # dataset_name = os.path.splitext(os.path.basename(input_file_path))[0]
    # data_init = pd.read_excel(input_file_path, sheet_name=sheet_name, 
    #                           index_col=index_col, skiprows=skiprows, skipfooter=skipfooter)
    
    # data_init.index = pd.to_datetime(data_init.index, format=data_format)

    data_init = pd.DataFrame()

    for file in input_files:
        df = pd.read_excel(os.path.join(input_folder, file + '.xlsx'), sheet_name=sheet_name, index_col=index_col, skiprows=skiprows, skipfooter=skipfooter)
        df.index = pd.to_datetime(df.index, format=data_format)

        df.columns = ['Average T', 'Max T', 'Min T', 'Precipitation']
        df.columns = df.columns.map(lambda x: file + '_' + x)
        df = df[~df.index.duplicated()]
        if data_init.empty:
            data_init = df
        else:
            data_init = pd.concat([data_init, df], axis=1)
    

    data_init_scaled, scaler = tsi.dataframe_scaler(df_input=data_init, 
                                                    time_column=time_column,  
                                                    preprocessing=preprocessing, 
                                                    index=index)

    names = data_init.columns.str.startswith(input_file)
    
    # vintage score
    vintage_init = calc_vintage_score(data_init.loc[:, names])

    data_missing = data_init_scaled.copy()
    specified_data_missing = data_missing.loc[(data_missing.index.month >= 8) & (data_missing.index.month <= 9), names]
    data_missing.loc[(data_missing.index.month >= 8) & (data_missing.index.month <= 9), names] = gap_generation_overlapping_perc(specified_data_missing, perc, length, seed)

    # transform to original
    # missing
    data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                     scaler=scaler,
                                                     time_column=time_column,
                                                     preprocessing=preprocessing,
                                                     index=index)

    # vintage score
    vintage_missing = calc_vintage_score(data_missing_orig.loc[:, names])
        
    if algorithm == 'Naive':
        params[algorithm] = {}
    
    results = {
        'dataset': input_file, 
        'gap_parameters': {
            'gap_type': 'overlap', 
            'perc': str(perc*100) + ' %', 
            'length': length,
            'seed': seed
        },
        'algorithm': algorithm,
        'algorithm_parameters': params[algorithm],
        'metrics': {},
        'vintage_score': {}
    }
    
    start_time = time.time()
    data_filled = tsi.run_imputation(missing = data_missing, algorithms=[algorithm], 
                                     time_column=time_column, params=params, preprocessing=preprocessing, 
                                     index=index, is_multivariate = is_multivariate)[algorithm]
    exec_time = (time.time() - start_time)
    data_filled.set_index(time_column, inplace=True)
    
    # metrics
    metrics = compute_metrics(data_init_scaled.loc[:, names], data_missing.loc[:, names], data_filled.loc[:, names])
    metrics['Execution Time'] = exec_time

    results['metrics'] = metrics

 
    # reverse transform imputed
    data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                    scaler=scaler,
                                                    time_column=time_column,
                                                    preprocessing=preprocessing,
                                                    index=index)

    # vintage score
    vintage_score = {'original': vintage_init, 'missing': vintage_missing}


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

    results['vintage_score'] = vintage_score

    return results
    