import pandas as pd 
# from tabulate import tabulate
import numpy as np 
#from pypots.utils.metrics import calc_mae, calc_mre, calc_rmse, calc_mse
from pypots.nn.functional import calc_mae, calc_mre, calc_rmse, calc_mse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import glob
import json
import math
from collections import defaultdict
from tenacity import RetryError


def handle_llm_output(df_missing, llm_responses):
    df_imputed = df_missing.copy()
    llm_responses_updated = defaultdict(list)
    
    number_of_none_prediction_from_LLM = 0
    number_of_none_prediction_from_LLM_list = []
    
    number_of_error_prediction_from_LLM = 0
    number_of_error_prediction_from_LLM_list = []
    
    for serie, list_of_imp_dates in llm_responses.items():
        for i in list_of_imp_dates:
            date, value = i[0], i[1]
            
            if isinstance(value, (RetryError, AttributeError)):
                print(date)
                lai_value = 0  # Replace with other imputation logic if needed
                number_of_error_prediction_from_LLM += 1
                number_of_error_prediction_from_LLM_list.append(str(value))
            elif isinstance(value, float) and math.isnan(value):
                print(date)
                lai_value = 0  # Replace with other imputation logic if needed
                number_of_none_prediction_from_LLM += 1
                number_of_none_prediction_from_LLM_list.append(value)
            elif not isinstance(value, (int, float)):
                print(date)
                lai_value = 0  # Replace with other imputation logic if needed
                number_of_none_prediction_from_LLM += 1
                number_of_none_prediction_from_LLM_list.append(value)
            else:
                lai_value = value
            
            llm_responses_updated[serie].append([date, lai_value])
            df_imputed.loc[df_imputed['time'] == date, serie] = lai_value
            #df_missing.loc[df_missing['time'] == date, serie] = lai_value
    
    print('Number of Errors collected in the LLMs response:', number_of_error_prediction_from_LLM, '\n',
          'Number of None collected from LLM:', number_of_none_prediction_from_LLM)
    
    return {
        "df_imputed": df_imputed,
        "llm_responses_updated": dict(llm_responses_updated),
        "number_of_none_prediction_from_LLM": number_of_none_prediction_from_LLM,
        "number_of_none_prediction_from_LLM_list": number_of_none_prediction_from_LLM_list,
        "number_of_error_prediction_from_LLM": number_of_error_prediction_from_LLM,
        "number_of_error_prediction_from_LLM_list": number_of_error_prediction_from_LLM_list,
    }


def missing_values_statistics(df):
    """
    Computes statistics regarding missing values for each time series (column) in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame with time series as columns.
    
    Returns:
        pd.DataFrame: A summary DataFrame with statistics about missing values.
    """
    stats = pd.DataFrame({
        "Total Values": df.shape[0],
        "Missing Values": df.isnull().sum(),
        "Missing Percentage (%)": (df.isnull().sum() / df.shape[0]) * 100,
        "Non-Missing Values": df.notnull().sum(),
        "First Missing Index": df.apply(lambda col: col[col.isnull()].index.min() if col.isnull().any() else None),
        "Last Missing Index": df.apply(lambda col: col[col.isnull()].index.max() if col.isnull().any() else None),
    })
    stats.index.name = "Time Series"
    return stats

def find_new_nan_mask(df1, df2):
    """
    Identifies new NaN values in df2 compared to df1 and returns a boolean mask.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame (may have additional NaN values).

    Returns:
    pd.DataFrame: A boolean mask where True indicates new NaN values in df2.
    """
    if not df1.shape == df2.shape:
        raise ValueError("Both dataframes must have the same shape.")
    
    # Create masks for NaN in both DataFrames
    nan_mask_df1 = df1.isna()
    nan_mask_df2 = df2.isna()
    
    # Identify new NaN values in df2 that are not in df1
    new_nan_mask = nan_mask_df2 & ~nan_mask_df1
    
    return new_nan_mask

def generate_custom_gaps_updated(data: pd.DataFrame,
                         gap_type: str = 'random',
                         miss_perc: float = 0.1,
                         gap_length: int = 3,
                         max_gap_length: int = 3,
                         max_gap_count: int = 5,
                         random_seed: int = 18931):
    """
    Generate custom gaps for the data, including the option to add "missing"
    gaps for all cases without overwriting existing NaN values.
    :param data: dataframe with the data
    :param gap_type: type of gap to be generated
    :param miss_perc: float between 0 and 1 which determines the size of the gap.
    :param gap_length: int between 0 and 100 that determines the size of the gap
    :param max_gap_length: maximum length of the gap
    :param max_gap_count: maximum number of gaps
    :param random_seed: Number for the RandomState
    :return: dataframe with the data with gaps
    """
    # Cast all columns to object type to support mixed values
    data_gaps = data.copy().astype(object)
    rng = np.random.RandomState(random_seed)

    N, M = data_gaps.shape

    if gap_type == 'random':
        col_indices = {col: i for i, col in enumerate(data_gaps.columns)}
        mask = np.zeros((len(data_gaps), len(data_gaps.columns)), dtype=bool)

        for col in data_gaps.columns:
            num_missing = rng.randint(1, max_gap_count + 1)
            starts = rng.randint(0, len(data_gaps) - max_gap_length, size=num_missing)
            ends = starts + rng.randint(1, max_gap_length + 1, size=num_missing)
            starts = np.clip(starts, 1, None)
            ends = np.clip(ends, None, len(data_gaps) - 1)
            for start, end in zip(starts, ends):
                for i in range(start, end):
                    if pd.notna(data_gaps.iloc[i, col_indices[col]]):
                        mask[i, col_indices[col]] = True

        data_gaps[mask] = "missing"

    elif gap_type == 'single':
        top_five_perc = N // 20
        max_pos = int(miss_perc * N + top_five_perc)

        for i in range(top_five_perc, max_pos):
            if pd.notna(data_gaps.iloc[i, 0]):
                data_gaps.iloc[i, 0] = "missing"

    elif gap_type == 'no_overlap':
        if M > N:
            M = N
        miss_size = N // M
        incr = 0

        for col_idx in range(M):
            for i in range(incr, incr + miss_size):
                if i < N and pd.notna(data_gaps.iloc[i, col_idx]):
                    data_gaps.iloc[i, col_idx] = "missing"
            incr = (incr + miss_size) % N

    elif gap_type == 'blackout':
        top_five_perc = N // 20
        max_pos = top_five_perc + gap_length

        for i in range(top_five_perc, max_pos):
            for col_idx in range(M):
                if pd.notna(data_gaps.iloc[i, col_idx]):
                    data_gaps.iloc[i, col_idx] = "missing"

    return data_gaps


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


def forecasting_metrics(y_true, y_pred):
    """
    Calculate common forecasting metrics: MAE, MSE, RMSE, MAPE, R-squared, MBD, sMAPE.
    
    Parameters:
    y_true (numpy array): The true values.
    y_pred (numpy array): The predicted values.
    
    Returns:
    dict: A dictionary with the calculated metrics.
    """
    
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Compute MSE (Mean Squared Error)
    mse = np.mean((y_true - y_pred)**2)
    
    # Compute RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # Compute MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Compute R-squared (R²)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Compute MBD (Mean Bias Deviation)
    mbd = np.mean(y_pred - y_true)
    
    # Compute sMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # Return all metrics as a dictionary
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R-squared': r_squared,
        'MBD': mbd,
        'sMAPE': smape
    }
    
    return metrics


def forecasting_metrics(y_true, y_pred):
    """
    Calculate common forecasting metrics: MAE, MSE, RMSE, MAPE, R-squared, MBD, sMAPE.

    Parameters:
    y_true (numpy array): The true values.
    y_pred (numpy array): The predicted values.

    Returns:
    dict: A dictionary with the calculated metrics.
    """
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute metrics using library functions
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # RMSE is not directly available in scikit-learn
    r_squared = r2_score(y_true, y_pred)
    
    # Compute MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Compute MBD (Mean Bias Deviation)
    mbd = np.mean(y_pred - y_true)
    
    # Compute sMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

    # Return all metrics as a dictionary
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R-squared': r_squared,
        'MBD': mbd,
        'sMAPE': smape
    }

    return metrics


def create_missing_values_with_tracking(df, column, window_size, num_missing, in_the_end=False):
    """
    Introduces missing values into specified columns (or all columns) of a pandas DataFrame and tracks the hidden values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str or list): The column(s) in which to introduce missing values. Use "all" to apply to all columns.
    window_size (int): The size of the sliding window.
    num_missing (int): The number of missing values to create within each sliding window.
    in_the_end (bool): If True, ensures one missing value is placed at the end of the window.

    Returns:
    tuple:
        - pd.DataFrame: A DataFrame with missing values introduced in the specified columns.
        - dict: A dictionary where the keys are column names and the values are Series containing the original values that were hidden.
    """
    # Determine the target columns
    if column == "all":
        target_columns = df.columns
    else:
        if isinstance(column, str):
            column = [column]  # Convert to list for consistency
        # Ensure all specified columns exist
        missing_columns = [col for col in column if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Column(s) {missing_columns} not found in the DataFrame.")
        target_columns = column

    # Ensure the sliding window and num_missing values are valid
    if window_size <= 0 or num_missing <= 0:
        raise ValueError("Window size and number of missing values must be positive integers.")
    if num_missing > window_size:
        raise ValueError("Number of missing values cannot exceed the window size.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    n_rows = len(df_copy)
    
    # Dictionary to store the original hidden values for each column
    hidden_values = {col: pd.Series(index=df_copy.index, dtype=df[col].dtype) for col in target_columns}

    # Apply the missing value process to each target column
    for col in target_columns:
        # Iterate through the DataFrame using the sliding window
        for start in range(0, n_rows, window_size):
            end = min(start + window_size, n_rows)
            indices = list(range(start, end))
            
            # If in_the_end is True, ensure that the last index in the window is missing
            if in_the_end:
                # Always include the last index of the window in the missing set
                missing_indices = {indices[-1]}
                # Randomly select the remaining missing indices from the window
                remaining_indices = np.random.choice(indices[:-1], size=min(num_missing - 1, len(indices) - 1), replace=False)
                missing_indices.update(remaining_indices)
                missing_indices = list(missing_indices)
            else:
                # Randomly select indices within the window to make missing
                missing_indices = np.random.choice(indices, size=min(num_missing, len(indices)), replace=False)
            
            # Save the original values
            hidden_values[col].loc[missing_indices] = df_copy.loc[missing_indices, col]
            
            # Assign NaN to the selected indices
            df_copy.loc[missing_indices, col] = np.nan
    
    return df_copy, pd.DataFrame(hidden_values)


def generate_report(df, date_column, columns='all', window=5):
    """
    Generates a text report based on the data in the specified columns.
    
    Parameters:
    - df: A pandas DataFrame with at least a Date column.
    - date_column: The name of the column containing date information.
    - columns: A list of column names for which the report will be created.
               If 'all' is provided, reports will be generated for all columns.
    - window: The sliding window size (optional, default is 5).
    
    Returns:
    - A dictionary where keys are the column names and values are the list of formatted strings describing the data.
    """
    # Initialize the report dictionary
    report = {}

    df[date_column] = pd.to_datetime(df[date_column])
    
    # Ensure the date column exists in the DataFrame
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in the DataFrame.")
    
    # Check if 'columns' is 'all' or a list of specific columns
    if columns == 'all':
        columns = df.columns.difference([date_column]).tolist()  # Exclude the date column
    
    # For each specified column, create a report
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        column_report = []
        
        # Iterate through the DataFrame and create a report for each date
        for index, row in df.iterrows():
            date_str = row[date_column].strftime('%d-%m-%Y')
            
            if pd.isna(row[column]):
                column_report.append(f"for the date {date_str} we are missing the number of {column}")
            else:
                column_report.append(f"for the date {date_str} the number of {column} were {row[column]}")
        
        # Add the report for the current column to the dictionary
        report[column] = column_report
    
    return report


def generate_report(df, window=5):
    """
    Generates a text report based on the sales data.
    
    Parameters:
    - df: A pandas DataFrame with columns 'Month' and 'Sales'.
    - window: The sliding window size (optional).
    
    Returns:
    - A list of formatted strings describing the sales data.
    """
    report = []
    
    for index, row in df.iterrows():
        date_str = row['Month'].strftime('%d-%m-%Y')
        
        if pd.isna(row['Sales']):
            report.append(f"for the date {date_str} we are missing the number of sales")
        else:
            report.append(f"for the date {date_str} the number of sales were {row['Sales']}")
    
    return report


def dict_to_markdown_table(results_dict):
    """
    Converts a dictionary of results into a markdown table.
    
    Parameters:
    - results_dict (dict): Dictionary where keys are model names and values are dictionaries 
                            with metric names and corresponding values.
                            
    Returns:
    - markdown_table (str): A markdown formatted table.
    """
    # Extract headers (columns) from the dictionary
    headers = ['Model'] + list(next(iter(results_dict.values())).keys())  # 'Model' + metric names

    # Initialize the markdown table
    markdown_table = '| ' + ' | '.join(headers) + ' |\n' 
    markdown_table += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n' 

    # Iterate over the dictionary and build rows for the markdown table
    for model, metrics in results_dict.items():
        row = [model] + [str(value) for value in metrics.values()]
        markdown_table += '| ' + ' | '.join(row) + ' |\n' 

    return markdown_table

def load_latest_json(folder_path):
    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        raise FileNotFoundError("No JSON files found in the specified folder.")
    
    # Get the most recently created or modified file
    latest_file = max(json_files, key=os.path.getctime)  # Use getctime for creation time
    
    # Load the JSON file
    with open(latest_file, 'r') as file:
        data = json.load(file)
    
    print(f"Loaded file: {latest_file}")
    return data



def timeseries_to_sentences(series_name, region, values, window_size):
    """
    Transform a time series into descriptive text sentences based on a sliding window.
    
    Args:
        series_name (str): Name of the time series (e.g., "LAI").
        region (str): Name of the region (e.g., "Denmark").
        values (list): List of time series values.
        window_size (int): Sliding window size.
    
    Returns:
        list: List of descriptive text sentences.

    Example: 

    # Example usage:
    series_name = "LAI"
    region = "Denmark"
    values = [1, 2, 3, 4, 5]
    window_size = 2

    output = timeseries_to_sentences(series_name, region, values, window_size)
    for sentence in output:
        print(sentence)
    """
    sentences = []
    
    # Ensure the window size is valid
    if window_size < 1 or window_size >= len(values):
        raise ValueError("Window size must be between 1 and the length of the series minus one.")
    
    for i in range(len(values) - window_size):
        window = values[i:i + window_size]
        next_value = values[i + window_size]
        
        # Construct the sentence
        window_text = ', '.join(f'"{v}"' for v in window)
        sentence = (
            f"The {series_name} values of {region} are {window_text} "
            f"and the next one is \"{next_value}\"."
        )
        sentences.append(sentence)
    
    return sentences


def timeseries_to_sentences_with_time(series_name, region, times, values, window_size):
    """
    Transform a time series with time information into descriptive text sentences based on a sliding window.
    
    Args:
        series_name (str): Name of the time series (e.g., "LAI").
        region (str): Name of the region (e.g., "Denmark").
        times (list): List of timestamps corresponding to the values.
        values (list): List of time series values.
        window_size (int): Sliding window size.
    
    Returns:
        list: List of descriptive text sentences including time information.

    Example: 
    # Example usage:
    series_name = "LAI"
    region = "Denmark"
    times = ["2024-01-01 12:00", "2024-01-01 13:00", "2024-01-01 14:00", "2024-01-01 15:00", "2024-01-01 16:00"]
    values = [1, 2, 3, 4, 5]
    window_size = 2

    output = timeseries_to_sentences_with_time(series_name, region, times, values, window_size)
    for sentence in output:
        print(sentence)
    """
    sentences = []
    
    # Ensure the inputs are valid
    if len(times) != len(values):
        raise ValueError("The length of the time list and values list must be the same.")
    if window_size < 1 or window_size >= len(values):
        raise ValueError("Window size must be between 1 and the length of the series minus one.")
    
    for i in range(len(values) - window_size):
        # Extract the sliding window for time and values
        window_times = times[i:i + window_size]
        window_values = values[i:i + window_size]
        next_time = times[i + window_size]
        next_value = values[i + window_size]
        
        # Construct the sentence
        window_text = ', '.join(
            f'"{value}" at "{time}"' for time, value in zip(window_times, window_values)
        )
        sentence = (
            f"The {series_name} values of {region} are {window_text} "
            f"and the next one is \"{next_value}\" at \"{next_time}\"."
        )
        sentences.append(sentence)
    
    return sentences



def generate_report(df, date_column, columns='all', window=5):
    """
    Generates a text report based on the data in the specified columns.
    
    Parameters:
    - df: A pandas DataFrame with at least a Date column.
    - date_column: The name of the column containing date information.
    - columns: A list of column names for which the report will be created.
               If 'all' is provided, reports will be generated for all columns.
    - window: The sliding window size (optional, default is 5).
    
    Returns:
    - A dictionary where keys are the column names and values are the list of formatted strings describing the data.
    """
    # Initialize the report dictionary
    report = {}

    df[date_column] = pd.to_datetime(df[date_column])
    
    # Ensure the date column exists in the DataFrame
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in the DataFrame.")
    
    # Check if 'columns' is 'all' or a list of specific columns
    if columns == 'all':
        columns = df.columns.difference([date_column]).tolist()  # Exclude the date column
    
    # For each specified column, create a report
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        column_report = []
        
        # Iterate through the DataFrame and create a report for each date
        for index, row in df.iterrows():
            date_str = row[date_column].strftime('%d-%m-%Y')
            
            if pd.isna(row[column]):
                column_report.append(f"for the date {date_str} we are missing the number of {column}")
            else:
                column_report.append(f"for the date {date_str} the number of {column} were {row[column]}")
        
        # Add the report for the current column to the dictionary
        report[column] = column_report
    
    return report


def split_list(lst, num_splits):
    """
    Splits a list into `num_splits` sublists.
    
    Parameters:
    - lst: The original list to split.
    - num_splits: The number of sublists to create.
    
    Returns:
    - A list of sublists.
    """
    if num_splits <= 0:
        raise ValueError("The number of splits must be a positive integer.")
    
    # Calculate the size of each split
    split_size = len(lst) // num_splits
    remainder = len(lst) % num_splits  # Items that will be distributed across the first few splits
    
    sublists = []
    start_idx = 0
    
    for i in range(num_splits):
        # Distribute the remainder items among the first few sublists
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        sublist = lst[start_idx:end_idx]
        sublists.append(sublist)
        start_idx = end_idx
    
    return sublists


def split_dataframe_by_rows(df: pd.DataFrame, n_rows_per_split: int, max_splits: int = None) -> list:
    """
    Splits the rows of a DataFrame into chunks of a specified number of rows per split.
    Optionally limits the total number of splits.

    Parameters:
    - df: pd.DataFrame - The DataFrame to split.
    - n_rows_per_split: int - The number of rows per split.
    - max_splits: int, optional - The maximum number of splits to return. If not provided, returns all splits.

    Returns:
    - list: A list of DataFrame splits.
    """
    if n_rows_per_split <= 0:
        raise ValueError("Number of rows per split must be greater than 0")
    
    n_total_rows = len(df)
    n_possible_splits = (n_total_rows + n_rows_per_split - 1) // n_rows_per_split  # Ceiling division to account for leftover rows
    
    # Apply max_splits limit if defined
    if max_splits is not None:
        n_possible_splits = min(n_possible_splits, max_splits)
    
    # Create splits
    splits = []
    for i in range(n_possible_splits):
        start_idx = i * n_rows_per_split
        end_idx = min(start_idx + n_rows_per_split, n_total_rows)  # Handle last split with fewer rows
        splits.append(df.iloc[start_idx:end_idx])
    
    return splits


# def missing_values_table_show(missing_stats, ascending_=False):
#     # Convert DataFrame to tabulate format
#     table = tabulate(missing_stats.sort_values(by=['Missing Percentage (%)'], ascending=ascending_), headers='keys', tablefmt='grid')

#     # Display a title
#     title = "Missing values of the original dataset"
#     print(f"\n{'='*55} {title} {'='*55}\n")
#     # Markdown("### Title of the Cell")
#     print(table)
