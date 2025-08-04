# from langchain_ollama import ChatOllama
# from langchain_core.pydantic_v1 import BaseModel
# from ollama_proxy.proxy import AnswerandJustification
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.pydantic_v1 import BaseModel
# from langchain_core.runnables import RunnableLambda

from utils.LLMs.ts_impute_functions import * 
from utils.LLMs.utility_functions import * 
from typing import Dict
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage)
from typing import List
import random 
import requests
import faiss
from datetime import datetime
import uuid
import sys
import os
import gc
from pydantic import BaseModel, Field


class AnswerandJustification(BaseModel):
    '''An answer to the user question '''

    #answer: str =  Field(description="The list of data and lai value pari that you were given but with imputed the missing value.")
    answer_value: float = Field(..., description="The float number")
    answer_date: str = Field(..., description="The requested date")
    answer_explanation: str = Field(..., description="Explaination of the missing values fullfillment")

# #######################################################################################################################################################################
# ###### RAG VERSION 6 (WITH DATE RANGE + without max_retries but with while loop searching until find the matches that are according to the rules and filters)
# #######################################################################################################################################################################

# Example: Convert a time series into a fixed-length embedding
def extract_features(series):
    """
    Extract basic statistical features from a 1D numerical array.

    This function computes a set of descriptive statistics commonly used for summarizing the distribution
    of values in a time series or any one-dimensional numerical data.

    Parameters:
    ----------
    series : array-like (1D)
        A NumPy array or list of numerical values.

    Returns:
    -------
    features : numpy.ndarray
        A NumPy array containing the following statistical features in order:
            - Mean of the series
            - Standard deviation of the series
            - Minimum value
            - Maximum value
            - 25th percentile (first quartile)
            - 50th percentile (median)
            - 75th percentile (third quartile)

    Notes:
    -----
    - Input should not contain NaN values. If missing data is present, it should be imputed beforehand.
    - These features are useful for feature engineering in machine learning models, especially for time series analysis.
    """

    return np.array([
        np.mean(series),
        np.std(series),
        np.min(series),
        np.max(series),
        np.percentile(series, 25),
        np.percentile(series, 50),
        np.percentile(series, 75)
    ])

# Segment a 1D time series into overlapping windows, including corresponding time ranges
def segment_time_series(ts, time_index, window_size, stride):
    """
    Segment a time series into overlapping or non-overlapping windows.

    This function splits a univariate time series into smaller subsequences (segments)
    of fixed size using a sliding window approach. For each segment, the corresponding
    time range is also returned based on the provided time index.

    Parameters:
    ----------
    ts : array-like
        The 1D array or list containing the time series values.
    
    time_index : array-like
        A list or array of datetime values (or strings) corresponding to the time dimension of the series.
        Must be the same length as `ts`.

    window_size : int
        The number of time steps in each segment (i.e., the window length).
    
    stride : int
        The step size to move the sliding window. Smaller stride means more overlap.

    Returns:
    -------
    segments : list of arrays
        A list containing each segmented window of time series values.
    
    time_ranges : list of arrays
        A list containing the corresponding time ranges for each segment.
        Each element aligns with the respective segment in `segments`.

    Notes:
    -----
    - The function does not pad the time series. Only complete windows are returned.
    - This is useful for time-series modeling tasks like forecasting, classification, or anomaly detection.
    """

    segments = []
    time_ranges = []  # List to hold date ranges per segment
    for start in range(0, len(ts) - window_size + 1, stride):
        segment = ts[start:start + window_size]
        segment_time_range = time_index[start:start + window_size]  # Corresponding time range for this segment
        segments.append(segment)
        time_ranges.append(segment_time_range)
    return segments, time_ranges

# Extract embeddings for segmented time series, including time ranges for each segment
def build_embeddings(time_series_data, time_series_time, window_size=50, stride=25):
    """
    Generate feature-based embeddings from a list of time series using a sliding window approach.

    This function processes multiple time series by segmenting each into windows and computing
    statistical features for each segment. It also tracks metadata and time ranges associated
    with each segment.

    Parameters:
    ----------
    time_series_data : list of array-like
        A list where each element is a univariate time series (e.g., list or 1D numpy array).

    time_series_time : list of array-like
        A list where each element contains the time indices corresponding to the values
        in `time_series_data`. Must match in length and order.

    window_size : int, optional (default=50)
        The number of time steps to include in each segment.

    stride : int, optional (default=25)
        The number of steps to move the sliding window for the next segment.

    Returns:
    -------
    all_embeddings : np.ndarray
        A 2D array where each row is the feature embedding of a segment.

    meta_info : list of dict
        A list of dictionaries containing metadata for each segment:
        - 'series_index': Index of the original time series.
        - 'segment_index': Index of the segment within the time series.

    time_ranges : list of array-like
        A list of time index ranges corresponding to each segment.

    Notes:
    -----
    - Each segment is processed using the `extract_features` function.
    - The embeddings can be used for downstream tasks such as clustering,
      classification, or similarity analysis between time segments.
    """

    all_embeddings = []
    meta_info = []  # To track which time series and which segment
    time_ranges = []  # To store the time ranges of each segment

    for ts_idx, (ts, time_index) in enumerate(zip(time_series_data, time_series_time)):
        segments, time_ranges_for_ts = segment_time_series(ts, time_index, window_size, stride)
        for seg_idx, (segment, time_range) in enumerate(zip(segments, time_ranges_for_ts)):
            embedding = extract_features(segment)
            all_embeddings.append(embedding)
            meta_info.append({
                "series_index": ts_idx,
                "segment_index": seg_idx
            })
            time_ranges.append(time_range)
    
    return np.array(all_embeddings), meta_info, time_ranges

# Build a FAISS index
def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search using L2 (Euclidean) distance.

    This function creates a flat index over the provided embeddings using
    the FAISS library, which enables efficient nearest neighbor search in high-dimensional space.

    Parameters:
    ----------
    embeddings : np.ndarray
        A 2D NumPy array of shape (n_samples, n_features) where each row represents
        a feature vector (embedding) of a time series segment or other data instance.

    Returns:
    -------
    index : faiss.IndexFlatL2
        A FAISS index built on the input embeddings, ready to be queried for nearest neighbors.

    Notes:
    -----
    - This function uses the `IndexFlatL2` class, which performs exact L2 distance computations.
    - It is suitable for small to medium datasets. For large-scale datasets, consider using
      `IndexIVFFlat`, `IndexHNSWFlat`, or other FAISS approximated indexing options.
    """

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 (Euclidean) distance
    index.add(embeddings)
    return index

# Build the full RAG structure
def build_rag(df_missing, window_size, stride):
    """
    Build a retrieval-augmented graph (RAG) structure from a dataframe containing missing time series data.

    The function performs the following steps:
    - Interpolates missing values linearly in the dataframe.
    - Extracts time series data and corresponding time indices.
    - Segments each time series into overlapping windows defined by the window size and stride.
    - Extracts feature embeddings from each segment.
    - Builds a FAISS index over the embeddings for efficient similarity search.
    - Creates mappings between time series column names and their indices.

    Parameters:
    ----------
    df_missing : pd.DataFrame
        DataFrame containing time series data with a 'time' column and one or more feature columns
        that may contain missing values.
    window_size : int
        The size of the sliding window used to segment each time series.
    stride : int
        The stride (step size) between consecutive windows during segmentation.

    Returns:
    -------
    time_series_data : np.ndarray
        2D array of shape (num_series, series_length) containing the interpolated and transposed time series data.
    column_names : pd.Index
        The column names corresponding to the different time series (features).
    time_series_time : list of np.ndarray
        List of arrays where each contains the timestamps corresponding to the time series data.
    name_to_index : dict
        Mapping from column names (time series names) to their integer indices.
    index_to_name : dict
        Reverse mapping from integer indices to column names.
    embeddings : np.ndarray
        Array of feature embeddings extracted from segmented time series windows.
    meta_info : list of dict
        Metadata for each embedding, including which series and segment it corresponds to.
    time_ranges : list of np.ndarray
        Corresponding time ranges for each segmented window used in embeddings.
    index : faiss.IndexFlatL2
        FAISS index built over the embeddings for fast nearest neighbor search.
    """

    df_missing = df_missing.interpolate(method='linear')
    time_series_data = df_missing.drop(columns=['time']).fillna(0).T.values
    column_names = df_missing.drop(columns=['time']).columns
    #time_series_time = df_missing['time'].values  # Extract the time column
    time_series_time = [df_missing['time'].values for _ in range(time_series_data.shape[0])]

    name_to_index = {name: idx for idx, name in enumerate(column_names)}
    index_to_name = {idx: name for name, idx in name_to_index.items()}

    embeddings, meta_info, time_ranges = build_embeddings(time_series_data, time_series_time, window_size, stride)
    index = build_faiss_index(embeddings)

    return time_series_data, column_names, time_series_time, name_to_index, index_to_name, embeddings, meta_info, time_ranges, index


def query_similar_time_series(query_series, index, meta_info, time_series_data,
                              column_names, time_ranges, stride, target_date, missing_date,
                              k, window_size, max_total_search=32):
    """
    Retrieve the top-k most similar time series segments to a given query series from a FAISS index,
    prioritizing segments that include a specified missing date. If insufficient matches are found
    containing the missing date, a fallback search ignoring the date constraint is performed.

    Parameters:
    ----------
    query_series : np.ndarray
        The 1D array representing the query time series segment for which similar segments are sought.
    index : faiss.IndexFlatL2
        A FAISS index built over embeddings of time series segments.
    meta_info : list of dict
        Metadata corresponding to each indexed embedding, including series and segment indices.
    time_series_data : np.ndarray
        Array containing the original time series data, shape (num_series, series_length).
    column_names : pd.Index or list
        Names of the time series corresponding to rows in time_series_data.
    time_ranges : list of np.ndarray
        List of arrays representing the timestamps corresponding to each indexed segment.
    stride : int
        The stride length used when segmenting the time series.
    target_date : str or pd.Timestamp
        The target date for contextual filtering (not directly used in this function but passed in).
    missing_date : str or pd.Timestamp
        The date around which retrieved segments should ideally include.
    k : int
        Number of similar segments to retrieve.
    window_size : int
        The size of each time series segment window.
    max_total_search : int, optional (default=32)
        Maximum number of candidates to consider progressively when searching for segments containing the missing date.

    Returns:
    -------
    retrieved_segments_info : list of dict
        List of metadata dictionaries describing the retrieved segments.
    retrieved_segments : list of np.ndarray
        List of the actual time series segments retrieved.
    distances : np.ndarray
        Array of distances corresponding to retrieved embeddings from the FAISS index.

    Notes:
    ------
    The search begins by filtering retrieved segments to those whose time ranges include the specified missing_date.
    If fewer than k such segments are found after searching up to max_total_search candidates, a fallback search
    ignoring the missing_date filter returns the top k closest segments by embedding distance.

    """

    query_embedding = extract_features(query_series).reshape(1, -1)

    retrieved_segments_info = []
    retrieved_segments = []
    all_indices_seen = set()

    missing_date = pd.to_datetime(missing_date)
    search_batch = k
    offset = 0

    # First attempt: search with missing_date filter
    while len(retrieved_segments_info) < k and offset < max_total_search:
        total_to_search = min(offset + search_batch, max_total_search)
        distances, indices = index.search(query_embedding, total_to_search)

        if offset >= len(indices[0]):
            break

        for idx in indices[0][offset:total_to_search]:
            if idx in all_indices_seen:
                continue
            all_indices_seen.add(idx)

            meta = meta_info[idx]
            segment_time_range = pd.to_datetime(time_ranges[idx])
            ts_idx = meta["series_index"]
            segment_index = meta["segment_index"]
            segment_start = segment_index * stride
            segment_end = segment_start + window_size
            segment = time_series_data[ts_idx, segment_start:segment_end]

            if missing_date in segment_time_range:
                retrieved_segments_info.append({
                    "series_name": column_names[ts_idx],
                    "series_index": ts_idx,
                    "segment_index": segment_index,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "segment_time_range": time_ranges[idx]
                })
                retrieved_segments.append(segment)

                if len(retrieved_segments_info) >= k:
                    break

        offset = total_to_search
        search_batch *= 2  # progressive search expansion

    # Fallback: ignore missing_date if not enough segments were found
    if len(retrieved_segments_info) < k:
        print(f"[Fallback] Found only {len(retrieved_segments_info)} valid segments with the missing_date. Falling back to best matches.")

        # Reset results
        retrieved_segments_info = []
        retrieved_segments = []
        all_indices_seen.clear()

        # Search again with initial k
        distances, indices = index.search(query_embedding, k)
        for idx in indices[0]:
            if idx in all_indices_seen:
                continue
            all_indices_seen.add(idx)

            meta = meta_info[idx]
            ts_idx = meta["series_index"]
            segment_index = meta["segment_index"]
            segment_start = segment_index * stride
            segment_end = segment_start + window_size
            segment = time_series_data[ts_idx, segment_start:segment_end]

            retrieved_segments_info.append({
                "series_name": column_names[ts_idx],
                "series_index": ts_idx,
                "segment_index": segment_index,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "segment_time_range": time_ranges[idx]
            })
            retrieved_segments.append(segment)

            if len(retrieved_segments_info) >= k:
                break

    return retrieved_segments_info, retrieved_segments, distances


#########################################################################################################
####################################### OTHER FUNCTIONS #################################################
#########################################################################################################

def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    print(state)
    return state

def impute_missing_values(series):
    """
    Impute missing values (NaNs) in a numerical array using the mean of the available (non-NaN) values.

    This function replaces all `np.nan` values in the input array with the mean of the non-missing values.
    It is intended for simple imputation in time series or numerical datasets.

    Parameters:
    ----------
    series : array-like (1D)
        A NumPy array or list of numerical values, where missing values are represented as `np.nan`.

    Returns:
    -------
    imputed_series : numpy.ndarray
        A NumPy array of the same shape as the input, with all `np.nan` values replaced by the mean of the non-NaN values.
    """

    return np.where(np.isnan(series), np.nanmean(series), series)

def find_previous_and_next_dates(n_of_sim_series, fin_arr, missing_date):
    """
    Identify the closest previous and next dates relative to a target date within a time series array.

    This function takes a target (missing) date and a 2D array representing time series data,
    and searches for the chronologically closest previous and next dates present in the array.
    It is primarily used to contextualize missing data points by identifying their immediate neighbors.

    Parameters:
    ----------
    n_of_sim_series : int
        An integer representing the index or identifier of the current iteration or series.
        This value is used for logging and debugging purposes only.

    fin_arr : array-like of shape (n_samples, 2)
        A 2D array where each row represents a date-value pair. Dates are expected as strings in the format '%Y-%m-%d'.

    missing_date : list, tuple, or array-like
        A one-element list (or similar structure) containing the target date string in the format '%Y-%m-%d'
        for which the closest previous and next available dates are to be found.

    Returns:
    -------
    previous_instance : numpy.ndarray of shape (1, 2)
        The row from `fin_arr` that contains the closest previous date (before the target). Returned as a NumPy array of shape (1, 2).
        If no previous date is found, returns an array with `None`.

    next_instance : numpy.ndarray of shape (1, 2)
        The row from `fin_arr` that contains the closest next date (after the target). Returned as a NumPy array of shape (1, 2).
        If no next date is found, returns an array with `None`.

    Notes:
    -----
    - Dates are assumed to be in string format 'YYYY-MM-DD'.
    - This function prints the results for debugging using the format: 
      "RAG -> ITERATION: <n_of_sim_series> PREVIOUS DATE: <previous_instance> NEXT DATE: <next_instance>".
    - Both returned arrays are explicitly typed as `dtype=object` to accommodate mixed-type values.
    """

    # Target date
    target_date = missing_date[0]
    # Convert target date to datetime
    target = datetime.strptime(target_date, '%Y-%m-%d')
    #target = target_date.strftime('%Y-%m-%d')

    # Initialize previous and next instances
    previous_instance = None
    next_instance = None

    # Iterate through the array
    for row in fin_arr:
        current_date = datetime.strptime(row[0], '%Y-%m-%d')
        
        if current_date < target:
            # Update previous_instance if current_date is closer
            if previous_instance is None or current_date > datetime.strptime(previous_instance[0], '%Y-%m-%d'):
                previous_instance = row
        elif current_date > target:
            # Update next_instance if current_date is closer
            if next_instance is None or current_date < datetime.strptime(next_instance[0], '%Y-%m-%d'):
                next_instance = row

    # Print results
    print("RAG ->", "ITERATION:", n_of_sim_series, "PREVIOUS DATE:", previous_instance, "NEXT DATE:", next_instance)

    return np.array([previous_instance], dtype=object), np.array([next_instance], dtype=object)


def get_window_data(df, date, prev_window_size=3, next_window_size=2):
    """
    Extracts a window of data from the DataFrame around the specified date.

    Args:
        df: The pandas DataFrame containing the data.
        date: The target date for the window.
        window_size: The number of rows to include before and after the target date.

    Returns:
        A pandas DataFrame containing the window of data.
    """

    # Convert 'time' column to datetime if it's not already
    df['time'] = pd.to_datetime(df['time'])

    # Find the index of the target date
    try:
        target_index = df[df['time'] == date].index[0]
    except IndexError:
        raise ValueError(f"Date '{date}' not found in the DataFrame.")

    # Calculate start and end indices for the window
    start_index = max(0, target_index - prev_window_size)
    end_index = min(len(df), target_index + next_window_size + 1)

    # Extract the window of data
    window_df = df.iloc[start_index:end_index].copy()

    return window_df

##############################################################
### improved version of generate_llm_examples
##############################################################
def generate_llm_examples(df_missing, create_examples, prompt_mode, num_examples=1, min_rows_required=1, target_date=None):
    """
    Generate example prompts and corresponding answers for a language model using time series data with missing values.

    This function selects time series columns from a DataFrame that meet a minimum threshold of available (non-NaN) values,
    and creates example prompts by removing a single data point from each selected series. The missing value is used
    to generate a prompt and an expected answer using the `create_examples` function.

    Parameters:
    ----------
    df_missing : pandas.DataFrame
        A DataFrame that includes a 'time' column and one or more series columns, some of which may contain missing values.

    create_examples : callable
        A function that takes three arguments (example, requested_date, prompt_mode) and returns a formatted
        language model prompt and an `AnswerandJustification` object.

    prompt_mode : str
        Determines how the prompts are formulated (e.g., 'simple', 'agriculture_enhanced').

    num_examples : int, optional (default=1)
        Number of examples to generate from valid columns in the DataFrame.

    min_rows_required : int, optional (default=1)
        Minimum number of non-missing values required in a column to be considered valid for prompt generation.

    target_date : datetime, optional
        Currently unused in the implemented logic, but can be used in the future to select rows near a specific date.

    Returns:
    -------
    examples : list of dict
        Each dictionary contains:
        - 'input': the generated prompt string
        - 'tool_calls': a list containing the structured answer object

    example_dicts : list of dict
        Each dictionary includes:
        - 'input': the generated prompt string
        - 'answer_value': float, the true value of the removed data point
        - 'answer_date': str or datetime, the date corresponding to the removed value
        - 'answer_explanation': str, textual justification for the prediction

    Raises:
    ------
    ValueError
        If no valid columns have enough non-NaN data to generate examples.

    Notes:
    -----
    - Only columns with at least `min_rows_required` non-NaN values are considered.
    - A single random data point is removed from each selected column to simulate a missing value.
    - There is a placeholder block for more advanced row selection based on `target_date`, but it is currently commented out.
    """
    
    examples = []
    example_dicts = []
    
    # Get the list of column names excluding the first one (time column)
    columns = df_missing.drop(columns=['time'])

    # Filter columns with enough non-NaN values
    valid_columns = [col for col in columns if df_missing[col].dropna().shape[0] >= min_rows_required]

    if not valid_columns:
        raise ValueError("No columns have enough non-NaN data to generate examples.")
    
    # Select random columns from valid ones
    selected_columns = random.sample(valid_columns, min(num_examples, len(valid_columns)))

    for random_column in selected_columns:
        fin_arr = df_missing[['time', random_column]].dropna().values
        
        if fin_arr.shape[0] == 0:
            continue  

        # Find the index closest to the given target_date
        if target_date is None:
            continue  # or raise an error if the date is mandatory


        ############################
        ### 1st way 
        ############################
        random_index = np.random.choice(fin_arr.shape[0])
        random_row = fin_arr[random_index]
        fin_arr = np.delete(fin_arr, random_index, axis=0)
        ############################

        ######################################
        ### 2nd way - experimental 
        ######################################
        # fin_arr['time'] = pd.to_datetime(fin_arr['time'])
        # #print('target date:', target_date)
        # fin_arr['abs_time_diff'] = fin_arr['time'].apply(lambda x: abs((x - target_date).total_seconds()))
        # closest_idx = fin_arr['abs_time_diff'].idxmin()
        # fin_arr['time'] = fin_arr['time'].astype('str')
        # random_row = fin_arr.loc[closest_idx].drop(labels='abs_time_diff').values
        # #print('random row:', random_row)
        # fin_arr = fin_arr.drop(columns='abs_time_diff')#.values
        # matched_index = fin_arr.index.get_loc(closest_idx)
        # total_rows = fin_arr.shape[0]
        # half_window = min(10, total_rows // 2)
        # start_index = max(0, matched_index - half_window)
        # end_index = min(total_rows, matched_index + half_window + 1)
        # fin_arr = fin_arr[start_index:end_index]
        # # Delete the row corresponding to the target
        # fin_arr = fin_arr.drop(index=fin_arr.index[matched_index - start_index], errors='ignore')
        # # Safety check: make sure random_row is still within bounds
        # if not (start_index <= matched_index < end_index):
        #     continue
        #######################################
    

        #print('fin array:', fin_arr)
        
        msgs, answer_list_length = create_examples(example=fin_arr, requested_date=random_row, prompt_mode=prompt_mode)
        examples.append({"input": msgs, "tool_calls": [answer_list_length]})
        example_dicts.append({'input':msgs, 'answer_value': answer_list_length.answer_value, 
                                              'answer_date': answer_list_length.answer_date, 
                                              'answer_explanation': answer_list_length.answer_explanation})
    
    return examples, example_dicts


def create_examples(example, requested_date, prompt_mode):
    """
    Generate a formatted prompt and corresponding answer explanation for a given time series example.

    This function constructs a prompt to estimate a missing value in a time series based on the provided 
    example data, the requested missing date, and the specified prompt mode. It also returns a structured 
    answer with justification using the `AnswerandJustification` class.

    Parameters:
    ----------
    example : list or array-like
        A time series array representing date-value pairs used as context for the prompt.

    requested_date : list or tuple
        A pair where:
        - requested_date[0]: str — the date for which the value is missing
        - requested_date[1]: float — the actual value (used in the reference answer)

    prompt_mode : str
        The mode to determine the style of prompt generation. 
        Supported values:
        - 'simple': A generic prompt referring to time series values.
        - 'agriculture_enhanced' or 'agriculture-enhanced': A more specific prompt referring to Leaf Area Index values for crops.

    Returns:
    -------
    tuple
        A tuple containing:
        - msgs: str — the formatted prompt string for the language model.
        - answer_list_length: AnswerandJustification — an object containing the expected answer, date, and explanation.

    Notes:
    -----
    - This function is primarily used to generate example prompts and their corresponding reference answers for training or evaluation.
    - The explanation text is hardcoded and illustrative based on seasonal or trend-based reasoning.
    """

    # print('HEREEE => prompt_mode:', prompt_mode)
    # print('example:', example)
    # print('requested_date:', requested_date)

    if prompt_mode == 'simple':
        msgs=f"The following array represents the time series values for certain dates: {example}. Estimate the float number that is missing for the requested date {[requested_date[0], np.nan]}, return float value, the date and give the explanation."
        answer_list_length = AnswerandJustification(
        msgs=f"The following array represents the time series values for certain dates: {example}. Estimate the float number that is missing for the requested date {[requested_date[0], np.nan]}, return float value, the date and give the explanation.",                   
        answer_date = requested_date[0], 
        answer_value = requested_date[1],
        answer_explanation = f'I guess that the missing value for that date is {requested_date[1]} based on the trend and the seasonality observed in the data.'
        )
        return msgs, answer_list_length

    elif prompt_mode == 'agriculture_enhanced' or prompt_mode == 'agriculture-enhanced':

        msgs=f"The following array represents the Leaf Area Index values for a crop for certain dates: {example}. Estimate the float number Leaf Area Index value for the requested date {[requested_date[0], np.nan]}, return float LAI value, the date and give the explanation."
        answer_list_length = AnswerandJustification(
        msgs=f"The following array represents the Leaf Area Index values for a crop for certain dates: {example}. Estimate the float number Leaf Area Index value for the requested date {[requested_date[0], np.nan]}, return float LAI value, the date and give the explanation.",                   
        answer_date = requested_date[0], 
        answer_value = requested_date[1],
        answer_explanation = f'I guess that the missing LAI value for that date is {requested_date[1]} because of the season'
        )
        return msgs, answer_list_length

        

def tool_example_to_messages(example: Dict) -> List[BaseMessage]:
    ''' 
    link: https://python.langchain.com/v0.1/docs/use_cases/query_analysis/how_to/few_shot/
    '''
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


##################################################
#######   Prompt creation function  ##############
##################################################

def prompt_creation(LLM,dataset_with_missing_values,serie,missing_date,crop_type,prompt_mode,multiple_choices=None):
    """
    Generate a user prompt string based on the provided dataset, prompt mode, and model configuration.

    This function creates natural language prompts for language models to estimate or choose predicted values 
    for missing data in time series, depending on the specified prompt mode and language model (LLM) type.

    Parameters:
    ----------
    LLM : str
        The name or type of the language model or prompt logic (e.g., 'hybrid_multiple_choice').
    
    dataset_with_missing_values : pandas.DataFrame
        The time series dataset containing a 'time' column and the target series with missing values.
    
    serie : str
        The name of the column in the dataset representing the target time series.

    missing_date : list of str
        The date (or list containing a single date) for which a prediction is required.

    crop_type : str
        The type of crop associated with the series (used in context-aware prompts).

    prompt_mode : str
        Defines the type of prompt to generate. For example, 'context_aware' includes crop type in prompt.

    multiple_choices : dict, optional
        A dictionary containing model predictions for the missing date.
        Required when `LLM` is a multiple-choice type (e.g., contains keys like 
        'linear interpolation prediction', 'SAITS prediction', etc.).

    Returns:
    -------
    str
        A formatted prompt string ready to be passed to the language model.

    Raises:
    ------
    ValueError
        If `multiple_choices` is not provided when required for a multiple-choice prompt type.
    """

    # Ensure requested_date is a single string (for formatting)
    requested_date = missing_date[0] if isinstance(missing_date, (list, tuple, np.ndarray)) else missing_date

    # Prepare base values
    template_values = {
        "value_type": "the Leaf Area Index" if prompt_mode == 'context_aware' else "the timeseries",
        "crop_type": f"for {crop_type}" if prompt_mode == 'context_aware' else "",
        "data_array": dataset_with_missing_values[['time', serie]].dropna().values,
        "requested_date": requested_date,
        "explanation": "an explanation",
    }

    #######################################
    ### Handle multiple-choice prompt v1
    #######################################
    if LLM == 'hybrid_multiple_choice':
        if multiple_choices is None:
            raise ValueError("multiple_choices must be provided for multiple-choice prompts.")

        template_values.update({
            "linear_interp": multiple_choices['linear interpolation prediction'],
            "saits": multiple_choices['SAITS prediction'],
            "timesnet": multiple_choices['TimesNET prediction'],
            "csdi": multiple_choices['csdi prediction'],
        })

        user_prompt_template = """The following array represents {value_type} values {crop_type} for certain dates: {data_array}. 
        Keep in mind that:
        - Linear interpolation prediction is {linear_interp}
        - SAITS prediction is {saits}
        - TimesNet prediction is {timesnet}
        - CSDI prediction is {csdi}
        Choose one of the given forecasts that you think is best for the date {requested_date}.
        DO NOT CHOOSE AND RETURN NEGATIVE FLOAT NUMBER.
        Return the chosen {value_type} number for the requested date and the date itself, and give {explanation}."""

        return user_prompt_template.format(**template_values)
    
    #######################################
    ### Handle multiple-choice prompt v2
    #######################################
    elif LLM == 'hybrid_multiple_choice_v2':
        if multiple_choices is None:
            raise ValueError("multiple_choices must be provided for multiple-choice prompts.")

        template_values.update({
            "linear_interp": multiple_choices['linear interpolation prediction'],
            "saits": multiple_choices['SAITS prediction'],
            "timesnet": multiple_choices['TimesNET prediction'],
            "csdi": multiple_choices['csdi prediction'],
        })

        user_prompt_template = """The following array represents {value_type} values {crop_type} for certain dates: {data_array}. 
        Keep in mind that:
        - Linear interpolation prediction is {linear_interp}
        - SAITS prediction is {saits}
        - TimesNet prediction is {timesnet}
        - CSDI prediction is {csdi}
        Choose one of the given forecasts that you think is best for the date {requested_date} or your own estimation if you think it will be better. 
        DO NOT CHOOSE AND RETURN NEGATIVE FLOAT NUMBER.
        Return the chosen {value_type} number for the requested date and the date itself, and give {explanation}."""

        return user_prompt_template.format(**template_values)

    #######################################
    ### Handle all other prompts 
    #######################################
    elif LLM not in ['hybrid_multiple_choice', 'hybrid_multiple_choice_v2']:
        user_prompt_template = """The following array represents {value_type} values {crop_type} for certain dates: {data_array}.
        Estimate the missing float value of the given {value_type} for the requested date {requested_date}. 
        Return the float value, the date, and provide {explanation}."""

        return user_prompt_template.format(**template_values)

def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)            # For single GPU
    # torch.cuda.manual_seed_all(seed)        # For multi-GPU
