import pandas as pd
import time
import numpy as np
from stelarImputation import imputation as tsi
import os


def custom_single_gap(df: pd.DataFrame, random_state: int = 2024) -> pd.DataFrame:
    """
    Introduce a single missing value in each column of a DataFrame, 
    ensuring that the missing values are not in the same position across columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    random_state (int): Random seed for reproducibility. Defaults to 2024.

    Returns:
    pd.DataFrame: DataFrame with a single missing value in each column.
    """
    df_miss = df.copy()
    state = np.random.RandomState(random_state)
    
    # Get the number of rows and columns
    num_rows, num_cols = df_miss.shape
    
    # Generate a random permutation of the row indices for each column
    random_rows = state.permutation(num_rows)[:num_cols]
    
    # Ensure that the random rows are unique (in case num_rows < num_cols)
    while len(set(random_rows)) < num_cols:
        random_rows = state.permutation(num_rows)[:num_cols]
    
    # Create a mask to set the selected values to NaN
    mask = np.zeros(df_miss.shape, dtype=bool)
    mask[random_rows, np.arange(num_cols)] = True
    
    # Set the selected values to NaN
    df_miss.values[mask] = np.nan
    
    return df_miss


def test_imputation_metrics(input_file_path: str, algorithm: str, params: dict, seed: int):

    preprocessing=False
    index=True
    is_multivariate = False

    dataset_name = os.path.splitext(os.path.basename(input_file_path))[0]
    data_init = pd.read_csv(input_file_path, index_col=0).sort_index().dropna(axis=0, how='all')

    # can be calculated at the start of every step if same image but different instances
    data_init_scaled, scaler = tsi.dataframe_scaler(df_input=data_init, 
                                                    time_column=data_init.index.name,  
                                                    preprocessing=preprocessing, 
                                                    index=index)

    data_missing = data_init_scaled.copy()
    data_missing = custom_single_gap(data_missing, seed)

    # transform to original - used in vintage score if different image then move it to the next for-loop
    # missing
    data_missing_orig = tsi.dataframe_inverse_scaler(df_input=data_missing,
                                                     scaler=scaler,
                                                     time_column=data_init.index.name,
                                                     preprocessing=preprocessing,
                                                     index=index)

        
    if algorithm == 'Naive':
        params[algorithm] = {}
    
    results = {
        'dataset': dataset_name, 
        'gap_parameters': {
            'gap_type': 'custom_single', 
            'seed': seed
        },
        'algorithm': algorithm,
        'algorithm_parameters': params[algorithm],
        'metrics': {}
    }
    
    start_time = time.time()
    data_filled = tsi.run_imputation(missing = data_missing, algorithms=[algorithm], 
                                     time_column=data_init.index.name, params=params, preprocessing=preprocessing, 
                                     index=index, is_multivariate = is_multivariate)[algorithm]
    exec_time = (time.time() - start_time)
    data_filled.set_index(data_init.index.name, inplace=True)
    
    # metrics
    metrics = tsi.compute_metrics(data_init_scaled, data_missing, data_filled)
    metrics['Execution Time'] = exec_time

    results['metrics'] = metrics

    # imputed
    data_filled_orig = tsi.dataframe_inverse_scaler(df_input=data_filled,
                                                    scaler=scaler,
                                                    time_column=data_init.index.name,
                                                    preprocessing=preprocessing,
                                                    index=index)

    return results
    