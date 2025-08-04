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
import sys
import os
from utils.LLMs.create_examples_and_RAG_functions import *
from itertools import chain
import time 
from tenacity import retry, stop_after_attempt, wait_fixed

from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage)
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


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

########################################################################################################################################
############################################# MAIN FEW SHOT RANDOM FUNCTION ############################################################
########################################################################################################################################

def few_shot_random_examples_main_loop(inference_engine, groq_api_key, llm_method, low_missing_values_combined_lai_df, df_missing, crop_type, prompt_mode, api_url, api_token, starting_date, end_date, num_of_examples, model_name, prefix=None):

    """
    Orchestrates the main loop for few shot prompting method with random examples, to estimate missing LAI values using an LLM.

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
    def make_request_few_shot_random_examples_main_loop_ollama(inference_engine, groq_api_key, llm_method, df_missing, serie, missing_date, crop_type, prompt_mode, api_url, api_token, few_shot_examples_text, model_name, prefix):
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
        
        # print('human_message_content:', human_message_content)
        # print('\n')

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
        
            human_message_content = HumanMessage(content=human_message_content)

            ## Invoke the model with LangChain
            structured_model = model.with_structured_output(AnswerandJustification)

            ##### few shot template ########
            example_prompt = ChatPromptTemplate.from_messages([
                    ("human", "{input}"),
                    ("ai", 
                    "answer_value: {answer_value}\n"
                    "answer_date: {answer_date}\n"
                    "answer_explanation: {answer_explanation}")
                ])

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=few_shot_examples_text
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data scientist answering structured questions."),
                few_shot_prompt,
                ("human", "{input}")
            ])

            ######################################
            ## Formatted prompt
            ######################################
            # formatted_prompt = prompt.format(input=request.messages[0]['content']) 
            formatted_prompt = prompt.format(input=human_message_content) 
            #print('formatted_prompt:', formatted_prompt)

            try:
                response = structured_model.invoke([HumanMessage(content=formatted_prompt)])
                response_dict = {"answer_value":response.answer_value, "answer_date":response.answer_date, "answer_explanation":response.answer_explanation}
                #print("Response:", response)
                status = '200-OK' if response is not None else '500-error'
            except Exception as e:
                print("Error during invocation:", str(e))
                status = str(e)
            
            elapsed_time = time.time() - start_time
            return response_dict, elapsed_time

        elif inference_engine == "ollama":

            human_message_content = [{"content":human_message_content}]

            lc_req = {
            "model": model_name, #"llama3.1",  # The model you want to use
            "inference_engine": inference_engine,
            "messages": human_message_content,
            "temperature": 0.0, 
            'chain_name': 'few_shot_prompting_random_examples', 
            'examples_msgs': few_shot_examples_text, 
            'groq_api_key': groq_api_key, 
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

    ###########################
    ## Main Loop 
    ###########################

    llm_responses = {}
    llm_responses_time_per_serie = {}

    for i, (serie, serie_values) in enumerate(df_missing.items()):

        if serie != 'time':

            llm_responses[serie] = []

            a = df_missing[['time', serie]]
            b = mask[['time', serie]]

            llm_responses_time_per_serie[serie] = []
            
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

                    ####################################################################################
                    ### Randomly choosen examples creation - using python function 
                    ####################################################################################            
                    examples, example_dicts = generate_llm_examples(df_missing, create_examples, prompt_mode=prompt_mode, num_examples=num_of_examples, target_date=pd.to_datetime(missing_date[0]))

                    ##########################################
                    ### Examples creation 
                    ##########################################
                    # example_msgs = [msg.dict() for ex in examples for msg in tool_example_to_messages(ex)]
                    # example_msgs = list(chain.from_iterable(tool_example_to_messages(ex) for ex in examples))
                    # example_msgs = [msg.dict() for msg in example_msgs if isinstance(msg, BaseMessage)]
                    # few_shot_examples_text = "\n\n".join(format_example(e) for e in example_dicts)
                    # print('few_shot_examples_text:', few_shot_examples_text)
                    # print('\n')

                    #print('example_dicts:', example_dicts)
                    # print('\n')

                    ######################################
                    ## # Invoke the model
                    ######################################

                    try:
                        # Invoke the model
                        llm_response, elapsed_time = make_request_few_shot_random_examples_main_loop_ollama(inference_engine, groq_api_key, llm_method, df_missing, serie, missing_date, crop_type, prompt_mode, api_url, api_token, example_dicts, model_name, prefix)

                        ####################################
                        ### handle response with JSON format
                        ####################################
                        # print('llm_response answer_value:', llm_response['answer_value'])
                        # print('llm_response answer_date:', llm_response['answer_date'])
                        # print('llm_response answer_explanation:', llm_response['answer_explanation'])
                        # if isinstance(low_missing_values_combined_lai_df, pd.DataFrame):
                        #     print('Real value:', low_missing_values_combined_lai_df.loc[low_missing_values_combined_lai_df["time"] == missing_date[0], serie].iloc[0])
                        response = [missing_date[0], llm_response['answer_value']]

                    except Exception as e:
                        # Handle exceptions gracefully
                        response = [missing_date[0], e]
                        print(f"Error: {e}")
                    finally:
                        # Always record the response and elapsed time
                        #print('response:', response)
                        llm_responses[serie].append(response)
                        llm_responses_time_per_serie[serie].append([missing_date[0], elapsed_time])

                    #print('\n')

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
