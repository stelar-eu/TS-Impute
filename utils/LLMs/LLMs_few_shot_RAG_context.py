# from langchain_ollama import ChatOllama
# from langchain_core.pydantic_v1 import BaseModel
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.pydantic_v1 import BaseModel
# from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage)
# from langchain_core.runnables import RunnableLambda
# from ollama_proxy.proxy import AnswerandJustification

from utils.LLMs.ts_impute_functions import * 
from utils.LLMs.utility_functions import * 
from typing import Dict
from typing import List
import random 
import requests
import sys
import os
import gc
from utils.LLMs.create_examples_and_RAG_functions import *
import time 
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_core.documents import Document

from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


@lru_cache(maxsize=1)
def get_cached_model_groq(model_name, provider, base_url, temperature):
    return init_chat_model(
        model=model_name,
        model_provider=provider,
        api_key=base_url,
        temperature=temperature,
        timeout=60
    )

@lru_cache(maxsize=1)
def get_cached_model(model_name, provider, base_url, temperature):
    return init_chat_model(
        model=model_name,
        model_provider=provider,
        base_url=base_url,
        temperature=temperature,
        timeout=60
    )

class AnswerandJustification(BaseModel):
    '''An answer to the user question '''

    #answer: str =  Field(description="The list of data and lai value pari that you were given but with imputed the missing value.")
    answer_value: float = Field(..., description="The float number")
    answer_date: str = Field(..., description="The requested date")
    answer_explanation: str = Field(..., description="Explaination of the missing values fullfillment")


################################################################################################################################################
############################################# MAIN FEW SHOT RAG-CONTEXT FUNCTION ###############################################################
################################################################################################################################################

def few_shot_RAG_context_main_loop(inference_engine, groq_api_key, llm_method, low_missing_values_combined_lai_df, df_missing, crop_type, prompt_mode, api_url, api_token, starting_date, end_date, num_of_examples, model_name, prefix=None):
    
    """
    Orchestrates the main loop for few shot prompting method with retrieval augmented geenerated context, to estimate missing LAI values using an LLM.

    This function iterates over missing data points in a (LAI) time series and uses a language model
    to estimate the missing values. It constructs context-specific prompts, communicates with an
    inference engine (e.g., Ollama or Groq), and integrates retry logic for robustness.

    Parameters:
        inference_engine (str): Inference backend to use, such as 'ollama' or 'groq'.
        groq_api_key (str): API key for the Groq backend (if applicable).
        llm_method (str): The method for generating prompts ('zero_shot_prompting').
        low_missing_values_combined_lai_df (pd.DataFrame): Reference dataset with fewer missing LAI values (or without missing values at all).
        df_missing (pd.DataFrame): DataFrame containing rows with missing LAI values to estimate.
        crop_type (str): Type of crop (e.g., 'maize', 'wheat') influencing prompt structure.
        prompt_mode (str): Mode for constructing the prompt (e.g., 'simple', 'agriculture-enhanced').
        api_url (str): Endpoint for the LLM API (e.g., http://localhost:11111/....).
        api_token (str): Token used for authenticating with the LLM API.
        starting_date (str or datetime): The start of the period to consider in the imputation part.
        end_date (str or datetime): The end of the period to consider in the imputation part.
        model_name (str): Name of the language model (e.g., 'llama3.1', 'mixtral-nemo').
        prefix (str): Main part of the filename, which also described the main parameters of the experiment.

    Returns:
        metrics(dict): A dictionairy of the metrics measured after the imputation when compared to the real values
    """

    ####################
    ### LLM prompting 
    ####################
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def make_request_few_shot_RAG_context_main_loop_ollama(inference_engine, groq_api_key, llm_method, df_missing, serie, missing_date, crop_type, prompt_mode, api_url, api_token, example_msgs, summary_docs, model_name, prefix):
        """
        Sends a prompt to the LLM inference engine and returns the model's response.

        This function builds a prompt using timeseries data and sends it to a remote LLM API endpoint (e.g., Ollama).
        The request includes authentication headers and inference configuration. The function includes retry logic 
        for fault tolerance.

        Parameters:
            inference_engine (str): The inference engine to use (e.g., 'ollama', 'groq').
            groq_api_key (str): API key used for Groq inference (if applicable).
            llm_method (str): Strategy used to construct the prompt (e.g., 'simple', 'few_shot').
            df_missing (pd.DataFrame): The DataFrame containing the row with the missing value.
            serie (pd.DataFrame): The time series in which the imputation is aimed, used to build the context.
            missing_date (str or datetime): The date for which the LAI value is missing.
            crop_type (str): Type of crop to tailor the prompt (e.g., 'maize', 'wheat').
            prompt_mode (str): Mode that controls prompt formatting (e.g.: 'simple', 'agriculture-enhanced').
            api_url (str): The URL of the LLM API to which the request will be sent.
            api_token (str): Token used for authenticating the API request.
            model_name (str): The specific model to use for inference (e.g., 'llama3.1', 'mixtral-nemo').
            prefix (str): Main part of the filename, which also described the main parameters of the experiment.

        Returns:
            tuple: 
                - response.json() (dict): The JSON response from the LLM inference API.
                - elapsed_time (float): Time taken (in seconds) to complete the request.
        """
        # Record the start time
        start_time = time.time()

        human_message_content = prompt_creation(llm_method, df_missing, serie, missing_date, crop_type, prompt_mode)
        ##human_message_content = [{"content":human_message_content}]

        #print('human_message_content:', human_message_content)

        ### here we treat the case of the user choice of Groq api or ollama installed in local premises 
        if inference_engine == 'groq' or inference_engine == 'ollama_local':

            api_key_ = groq_api_key if inference_engine == 'groq' else api_url if inference_engine == 'ollama_local' else ValueError(f"Unknown inference engine: {inference_engine}")
            #print('api key:', api_key_)
            
            if inference_engine == 'groq':

                model = get_cached_model_groq(
                    model_name,
                    inference_engine,
                    api_key_,
                    0.0
                )

            elif inference_engine == 'ollama_local':
                
                model = get_cached_model(
                    model_name,
                    inference_engine,
                    api_key_,
                    0.0
                )
        
            human_message_content = [HumanMessage(content=human_message_content)]

            ## Invoke the model with LangChain
            structured_model = model.with_structured_output(AnswerandJustification)

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a data scientist specialized in ecological timeseries modeling."),
                HumanMessage(content="Take into consideration the following information: {context}"),
                MessagesPlaceholder(variable_name="msgs")
            ])
            #############################################################################
            ## Chain generation and contextual info generation using the retrieved series
            #############################################################################
            #print('summary docs:',  summary_docs)
            chain = {"context": lambda x: summary_docs, "msgs": RunnablePassthrough()} | RunnableLambda(inspect) | prompt | structured_model

            try:
                response = chain.invoke(human_message_content)
                response_dict = {"answer_value":response.answer_value, "answer_date":response.answer_date, "answer_explanation":response.answer_explanation}
                #print("Response:", response)
                status = '200-OK' if response is not None else '500-error'
            except Exception as e:
                print("Error during invocation:", str(e))
                status = str(e)

            elapsed_time = time.time() - start_time
            return response_dict, elapsed_time
        
        elif inference_engine == 'ollama':

            human_message_content = [{"content":human_message_content}]

            lc_req = {
            "model": model_name, # "deepseek-r1:8b",  # The model you want to use
            "inference_engine": inference_engine, 
            "groq_api_key": groq_api_key,
            "messages": human_message_content,
            "temperature": 0.0, 
            'chain_name': 'few_shot_prompting_rag_contextual_info', 
            #'example_msgs': example_msgs, 
            'dataset': None, #df_missing.replace({np.nan:'-10000'}).to_dict(), 
            'documents': summary_docs,
            'prefix': prefix, 
            'requested_date': missing_date[0] if isinstance(missing_date, (list, tuple, np.ndarray)) else missing_date, 
            'serie': serie
            }

            # Send the POST request with Authorization header and request data
            response = requests.post(
                api_url,
                json=lc_req,  # This is the payload containing messages, model, etc.
                headers={"Authorization": f"Bearer {api_token}"}  # API key header
            )

            elapsed_time = time.time() - start_time
            return response.json(), elapsed_time
    
    #######################################################
    # if 'Unnamed: 0' in df_missing.columns:
    #     df_missing = df_missing.drop(columns=['Unnamed: 0'])
    # if low_missing_values_combined_lai_df:
    #     if 'Unnamed: 0' in low_missing_values_combined_lai_df.columns:
    #         low_missing_values_combined_lai_df = low_missing_values_combined_lai_df.drop(columns=['Unnamed: 0'])
    #######################################################

    #######################################################
    
    if isinstance(low_missing_values_combined_lai_df, pd.DataFrame):
        mask = find_new_nan_mask(df1=low_missing_values_combined_lai_df, df2=df_missing)
        #print('low_missing_values_combined_lai_df exists')
    else:
        #print('low_missing_values_combined_lai_df is missing')
        mask = df_missing.isna()
    #######################################################

    ###########################
    ## RAG
    ###########################
    window_size_ = 10
    stride_ = 10
    time_series_data, column_names, time_series_time, name_to_index, index_to_name, embeddings, meta_info, time_ranges, index = build_rag(df_missing, window_size=window_size_, stride=stride_)

    ###########################
    ## Main Loop 
    ###########################

    llm_responses = {}
    llm_responses_time_per_serie = {}
    llm_explanations = {}
    distances_dict = {}

    for i, (serie, serie_values) in enumerate(df_missing.items()):

        if serie != 'time':

            llm_responses[serie] = []

            a = df_missing[['time', serie]]
            b = mask[['time', serie]]

            llm_responses_time_per_serie[serie] = []

            distances_dict[serie] = []
            
            for idx, missing_date in enumerate(a[b.iloc[:, 1]].values):

                #if (starting_date =='None' or pd.to_datetime(missing_date[0]) < pd.to_datetime(end_date)) and (end_date =='None' or pd.to_datetime(missing_date[0]) > pd.to_datetime(starting_date)):
                if (
                    (pd.isna(starting_date) or starting_date == 'None' or pd.to_datetime(missing_date[0]) < pd.to_datetime(end_date)) and
                    (pd.isna(end_date) or end_date == 'None' or pd.to_datetime(missing_date[0]) > pd.to_datetime(starting_date))
                ):
                    
                    # print('starting_date:', starting_date)
                    # print('end_date:', end_date)
                    # print('missing_date:', missing_date)
                    
                    # print('\n')
                    # print('ITERATION:', i, 'SERIE:', serie, 'MISSING DATE:', missing_date)

                    ###########################
                    ## RAG usage
                    ###########################
                    query_series = df_missing[serie].interpolate(method='linear').dropna().values.flatten()


                    ##############
                    ### New RAG 
                    ##############
                    ############################################
                    ####### RAG improved with dates ############
                    ############################################ 
                    k = num_of_examples
                    start_period = pd.to_datetime(missing_date[0]) - pd.Timedelta(weeks=2)  # Two weeks before
                    end_period = pd.to_datetime(missing_date[0]) + pd.Timedelta(weeks=2)    # Two weeks after
                    date_range = [start_period, end_period]
                    retrieved_segments_info, retrieved_segments, distances = query_similar_time_series(query_series, index, meta_info, time_series_data,
                                              column_names, time_ranges, stride_, date_range, missing_date[0],
                                              k, window_size_)
                    # Extract the series names and segment indices (for logging or further processing)
                    similar_series_names = [info["series_name"] for info in retrieved_segments_info]
                    similar_series_index = [info["series_index"] for info in retrieved_segments_info]
                    segment_indices = [info["segment_index"] for info in retrieved_segments_info]
                    segment_starts = [info["segment_start"] for info in retrieved_segments_info]
                    segment_ends = [info["segment_end"] for info in retrieved_segments_info]
                    segment_time_ranges = [info["segment_time_range"] for info in retrieved_segments_info]

                    # # Print the retrieved segments and their indices, along with the time range
                    # print('Similar segments:', retrieved_segments)
                    # print('similar_series_index:', similar_series_index)
                    # print('segment_indices:', segment_indices)
                    # print('Segment starts:', segment_starts)
                    # print('Segment ends:', segment_ends)
                    # print('Segment time ranges:', segment_time_ranges)  # Print the time ranges for the segments

                    examples = []
                    context_examples_list = []
                    context_examples = {}
                    num_of_examples_count = 0

                    for j in range(len(similar_series_names)):
                        
                        fin_arr = df_missing[['time', similar_series_names[j]]].dropna().values

                        ## try to find the same date in the data 
                        context_ex = fin_arr[fin_arr[:, 0]==missing_date[0]]

                        previous_instance, next_instance = find_previous_and_next_dates(j, fin_arr, missing_date)

                        ## if the same date was not found in the data 
                        if len(context_ex) == 0:
                            #print('EXAMPLE MESSAGES -> Didnt find the same date in the available data')
                            # Define the date to match
                            date_to_match = missing_date[0]

                            # Convert the target date to a datetime object
                            target_date = datetime.strptime(date_to_match, '%Y-%m-%d')

                            # Convert the first column of dates to datetime objects
                            dates = np.array([datetime.strptime(date, '%Y-%m-%d') for date in fin_arr[:, 0]])

                            # Find the index of the closest date
                            date_diffs = np.abs(dates - target_date)
                            closest_index = np.argmin(date_diffs)

                            # Get the row with the closest date
                            closest_row = fin_arr[closest_index]
                            context_ex = fin_arr[fin_arr[:, 0]==closest_row[0]]

                        ## if the same date was found in the data 
                        else:
                            #print('EXAMPLE MESSAGES -> Found the same date in the available data')
                            ## choose the instance that is similar in date and exlude it in order to create example 
                            closest_row = context_ex[0]
                            # Delete rows where the first column matches missing_date[0]
                            fin_arr = fin_arr[fin_arr[:, 0] != missing_date[0]]

                        msgs, answer_list_length = create_examples(example=fin_arr, requested_date=closest_row, prompt_mode=prompt_mode)
                        examples.append({"input": msgs, "tool_calls": [answer_list_length]})
                        
                        ########################################################
                        ## Build the context - this is for the similar series
                        ########################################################
                        formatted_rows = [f"for date {row[0]} for a similar crop the LAI value is {row[1]}" for row in context_ex]
                        formatted_rows_prev = [f"for previous date {row[0]} for a similar crop the LAI value is {row[1]}" for row in previous_instance if not None in previous_instance]
                        formatted_rows_next = [f"for next date {row[0]} for a similar crop the LAI value is {row[1]}" for row in next_instance if not None in next_instance]
                        formatted_rows = formatted_rows_prev + formatted_rows + formatted_rows_next
                        result_str =  ",".join(formatted_rows) #"[" +   + "]"
                        context_ex = str([[row[0], row[1]] for row in context_ex][0])
                        context_examples_list.append(result_str)
                    
                    #######################################################################################################################################
                    ## newly added - Build the context - previous and next dates and LAI values given to Context for the timeserie itself 
                    #######################################################################################################################################
                    exception_rows = a[a['time']==missing_date[0]]

                    # Drop rows with NaN in the rest of the DataFrame
                    df_dropped = a.dropna()

                    # Combine the exception rows back into the DataFrame
                    c = pd.concat([df_dropped, exception_rows]).sort_index()
                    c = c.reset_index().drop(columns='index')
                    #print(exception_rows, c.iloc[c[c['time']==missing_date[0]].index[0]-1], c.iloc[c[c['time']==missing_date[0]].index[0]+1])
                    try:
                        # Get the previous date and value
                        prev_date = c.iloc[c[c['time'] == missing_date[0]].index[0] - 1]['time']
                        prev_lai = c.iloc[c[c['time'] == missing_date[0]].index[0] - 1][serie]
                        prev_info = f'Remember that the Leaf Area Index value for the date {prev_date} is {prev_lai}'
                    except IndexError:
                        # If an IndexError occurs, handle it gracefully (e.g., set None or print a message)
                        prev_date = "missing"
                        prev_lai = "missing"
                        #print(f"Previous data for missing date {missing_date[0]} not found.")
                        prev_info = ''

                    try:
                        # Get the next date and value
                        next_date = c.iloc[c[c['time'] == missing_date[0]].index[0] + 1]['time']
                        next_lai = c.iloc[c[c['time'] == missing_date[0]].index[0] + 1][serie]
                        next_info = f'the Leaf Area Index value for the date {next_date} is {next_lai}'
                        
                    except IndexError:
                        # If an IndexError occurs, handle it gracefully (e.g., set None or print a message)
                        next_date = "missing"
                        next_lai = "missing"
                        #print(f"Next data for missing date {missing_date[0]} not found.")
                        next_info = ''

                    
                    retr_prev_next = prev_info + next_info
                    context_examples_list.append(retr_prev_next)
                    ###########################

                    ###########################
                    ## Docs linked to summaries
                    ###########################
                    doc_ids = [str(uuid.uuid4()) for _ in context_examples_list]
                    id_key = "doc_id"
                    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[j]}) for j, s in enumerate(context_examples_list)]

                    serialized_docs = [
                        {"metadata": doc.metadata, "page_content": doc.page_content}
                        for doc in summary_docs
                    ]

                    #print('serialized_docs:', serialized_docs)

                    example_msgs = [msg.dict() for ex in examples for msg in tool_example_to_messages(ex)]
                    #print('EXAMPLE MESSAGES:', example_msgs)

                    ######################################
                    #### Invoke the model
                    ######################################

                    try:
                        # Invoke the model
                        llm_response, elapsed_time = make_request_few_shot_RAG_context_main_loop_ollama(inference_engine, groq_api_key, llm_method, df_missing, serie, missing_date, crop_type, prompt_mode, api_url, api_token, example_msgs, serialized_docs, model_name, prefix)

                        ####################################
                        ### handle response with JSON format
                        ####################################
                        # print('llm_response answer_value:', llm_response['answer_value'])
                        # print('llm_response answer_date:', llm_response['answer_date'])
                        # print('llm_response answer_explanation:', llm_response['answer_explanation'])
                        # if isinstance(low_missing_values_combined_lai_df, pd.DataFrame):
                        #     print('Real value:', low_missing_values_combined_lai_df.loc[low_missing_values_combined_lai_df["time"] == missing_date[0], serie].iloc[0])
                        response = [missing_date[0], llm_response['answer_value']]
                        #print('similarity distances:', distances)

                    except Exception as e:
                        # Handle exceptions gracefully
                        response = [missing_date[0], e]
                        print(f"Error: {e}")
                    finally:
                        # Always record the response and elapsed time
                        #print('RESPONSE:', response)
                        llm_responses[serie].append(response)
                        llm_responses_time_per_serie[serie].append([missing_date[0], elapsed_time])

                    # # Explicitly trigger garbage collection every N iterations
                    # if (i + 1) % 50 == 0:  # Adjust the frequency (e.g., every 50 or 100 iterations)
                    #     gc.collect()
                    #     print("Explicit garbage collection triggered.")
                    #     # Trigger restart
                    #     server_response = requests.post("http://localhost:9067/restart-ollama")
                    #     print(server_response.json())

                else:
                    pass

                time.sleep(4)

    #############################
    ### Handling the LLMs ouput 
    #############################
    after_handling_output = handle_llm_output(df_missing, llm_responses)
    df_imputed = after_handling_output['df_imputed']
    llm_responses_updated = after_handling_output['llm_responses_updated']
    number_of_error_prediction_from_LLM = after_handling_output['number_of_error_prediction_from_LLM']
    number_of_error_prediction_from_LLM_list = after_handling_output['number_of_error_prediction_from_LLM_list']
    number_of_none_prediction_from_LLM = after_handling_output['number_of_none_prediction_from_LLM']
    number_of_none_prediction_from_LLM_list = after_handling_output['number_of_none_prediction_from_LLM_list']

    # print('Number of Errors collected in the LLMs response:', number_of_error_prediction_from_LLM, '\n',
    #         'Number of None collected from LLM:', number_of_none_prediction_from_LLM)
    

    #################################
    ### Return results & imputed df
    #################################
    if isinstance(low_missing_values_combined_lai_df, pd.DataFrame):

        if (pd.isna(starting_date) or starting_date == 'None') and (pd.isna(end_date) or end_date == 'None'):

            results = compute_metrics(original_df=low_missing_values_combined_lai_df.drop(columns=['time']), 
                        missing_df=df_missing.drop(columns=['time']), 
                        imputed_df=df_imputed.fillna(0).drop(columns=['time']))

            #print(results)
            
            return results, df_imputed
    
        else:
            filtered_low_missing_values_combined_lai_df = low_missing_values_combined_lai_df[(low_missing_values_combined_lai_df['time'] >= starting_date) & (low_missing_values_combined_lai_df['time'] <= end_date)]
            filtered_df_missing = df_missing[(df_missing['time'] >= starting_date) & (df_missing['time'] <= end_date)]
            filtered_df_imputed = df_imputed[(df_imputed['time'] >= starting_date) & (df_imputed['time'] <= end_date)]

            results = compute_metrics(original_df=filtered_low_missing_values_combined_lai_df.drop(columns=['time']), 
                        missing_df=filtered_df_missing.drop(columns=['time']), 
                        imputed_df=filtered_df_imputed.fillna(0).drop(columns=['time']))

            #print(results)
            
            return results, df_imputed
    
    else:
        return 'None', df_imputed