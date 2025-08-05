# Welcome to the KLMS Tool version of TS-Impute
TS-Impute is a library implemented in python and c# for time series missing value imputation.    
It includes a plethora of state of the art missing value imputation models (e.g. deep learning such as SAITS, TimesNet, etc.), statistical-methods (e.g. DynaMMO, SoftImpute, etc.) and llm-based-methods (e.g. zero_shot_prompting, few_shot_prompting, etc.)
# Instructions

## Input
To run TS-Impute, via STELAR KLMS API, the following input JSON structure is expected:
```json
{
  "process_id": "<PROCESS_ID>",
  "inputs": {
    "missing": [
        "<RESOURCE_UUID_1>"
    ],
    "ground_truth": [
        "<RESOURCE_UUID_2>"
    ]
  },
  "outputs": {
    "imputed_timeseries": {
        "url":"s3://<BUCKET_NAME>/temp/imputed_timeseries{.csv, .xlsx or .xls}",
        "resource": {
            "name": "Imputed data of <RESOURCE_UUID_1>",
            "relation": "imputed",
            "format": "{csv, xlsx or xls}"
        }
    }
  },
  "datasets":{
    "d0": "<DATASET_UUID>" 
  },
  "parameters": {
        "imp_method": "naive",
        "...": "additional parameters can be defined here"
  },
  "secrets": {
      "endpoint": "Endpoint/of/LLM",
      "token": "Token/for/Endpoint"
  }
}
```

### Input
- **`missing`** *(str, required)*  
  Path to the **csv**, **xlsx** or **xls** file containing the data with missing values.

- **`ground_truth`** *(str, optional, default=None)*  
  Path to the **csv**, **xlsx**, or **xls** file containing the ground truth values.   
  If not provided, the imputation output will not include evaluation metrics.

### Parameters common for all cases

- **`imp_method`** *(str, required, default="naive")*    
  Specifies the model or method used for imputing missing values.  
  At least one method must be provided.  
  The user can choose from the following options:  
  `"iterativesvd"`, `"rosl"`, `"cdmissingvaluerecovery"`, `"ogdimpute"`, `"nmfmissingvaluerecovery"`, `"dynammo"`, `"grouse"`, `"softimpute"`,  
  `"pca_mme"`, `"svt"`, `"spirit"`, `"tkcm"`, `"linearimpute"`, `"meanimpute"`, `"zeroimpute"`, `"naive"`, `"saits"`, `"brits"`, `"csdi"`,  
  `"usgan"`, `"mrnn"`, `"gpvae"`, `"timesnet"`, `"nonstationary_transformer"`, `"autoformer"`, `"llm_based"`.

  **NOTE:** For the `"llm_based"` option, the user can specify which method to use by setting the parameter [`llm_method`](#llm_method).


- **`sep`** *(str, optional, default=",")*  
  The delimiter used to separate values in the tabular and timeseries data.    
  If not specified, the default is comma(,).

- **`header`** *(int or bool, optional, default=0)*  
  The row number that contains the header of the tabular and timeseries  data.   
  If header is true then header=0 and if header is false then header=None.   
  If not provided, the default is 0.  

- **`time_column`** *(str, optional, default="time")*    
  Specify the name of the column that contains time values. If not specified, the default is "time".

- **`time_format`** *(str, optional, default="%Y-%m-%d %H:%M:%S")*    
  Specify format for the datetime column. If not specified, the default is "%Y-%m-%d %H:%M:%S".

### Parameters for all models and statistical-methods

- **`hyperparameter_tuning`** *(bool, optional, default=false)*  
  Indicates whether to perform hyperparameter tuning on the specified **imp_method**.   
  **_WARNING:_** Enabling hyperparameter tuning can significantly increase the execution time for imputation.

- **`default`** *(bool, optional, default=true)*  
  If `True`, each specified **imp_method** uses its own standard default parameters. When hyperparameter tuning is enabled, a different predefined set of parameters is tested for each specified **imp_method**.  
  If `False`, the user must specify the parameters manually (see the [`params`](#params) parameter).

- **`is_multivariate`** *(bool, optional, default=false)*  
  If `True`, the data is a single multivariate time series.  
  If `False`, there are multiple time series each with a single feature.

- **`n_trials`** *(int, optional, default=20)*  
  Number of trials for hyperparameter tuning. Used only if `hyperparameter_tuning` is `True`. The default is 20.

- **`tuning_top_k_fewer_missing`** *(int, optional, default=1000)*  
  Number of columns with the fewest missing values to use during hyperparameter tuning. If not specified, the default is 1000.  

- **`params`** *(dict, optional, default={})*   
  Parameters for the chosen **imp_method**.  
  Used only if [`default`](#default) is `False`; in that case, the user must specify these parameters manually. When hyperparameter tuning is enabled, the required `params` dictionary differs accordingly.  
  Examples of default parameters for each **imp_method**, including cases when hyperparameter tuning is enabled, can be found [here](https://github.com/stelar-eu/TS-Impute/blob/main/utils/stelarImputation/stelarImputation/default.py).   
  **_EXAMPLE (dynaMMO):_** 

  If [`hyperparameter_tuning`](#hyperparameter_tuning) is `False` then:
  ```json
  "params" : {
    "dynammo": {
        "H": 5,  
        "max_iter": 100,  
        "fast": true
    }
  }
  ```

  If [`hyperparameter_tuning`](#hyperparameter_tuning) is `True` then:
  ```json
  "params" : {
    "dynammo": {
        "H": [
            1,
            25
        ],
        "max_iter": [
            100
        ],
        "fast": [
            false,
            true
        ]
    }
  }
  ```
  
### Parameters for llm-based methods
- **`inference_engine`** *(str, optional, default="groq")*   
  The inference engine that is used for the usage of LLM-based methods. The default option is via the GROQ api (which requires the user create credentials before using this option).
 
- **`prompt_mode`** *(str, optional, default="simple")*    
  Mode that controls prompt formatting (e.g.: 'simple', 'context_aware'). In the "simple" version the prompt that is given to the LLM is domain agnostic (e.g.: The following array represents the **time series values** for certain dates...),
  while in the "context_aware" mode the prompt is domain-specific, tailored to the Leaf area index values (e.g.: The following array represents **the Leaf Area Index values** for **maize** for certain dates)
 
- **`crop_type`** *(str, optional, default=None)*  
  The type of crop associated with the series (used only in context-aware prompts)
 
- **`num_of_examples`** *(int, optional, default=1)*   
  The number of examples used in retrieval augmented generation.   
  This is used in few-shot prompting methods (few-shot random, few-shot RAG and few-shot context)
 
-  **`model_name`** *(str, optional, default="llama-3.1-8b-instant")*   
  The name of the LLM model from GROQ api that will be used
 
-  **`starting_date`** *(str, optional, default="None")*   
  The starting_date field is used in the case that the user wants to only receive inputation for a specific subset of the dataset, and this denotes the starting date of the subset period (of the dataset)
 
-  **`end_date`** *(str, optional, default="None")*   
  The end_date field is used in the case that the user wants to only receive inputation for a specific subset of the dataset, and this denotes the end date of the subset period (of the dataset)
 
-  **`llm_method`** *(str, optional, default="zero_shot_prompting")*   
  The method that will be used. Available methods are:
   - `zero_shot_prompting`: Zero shot prompting method iterates over the dataset (per column), and for each missing value in each column (time-serie), constructs the related prompt and feeds it to the selected LLM model.
   - `few_shot_prompting_random_examples`: Few shot prompting at random method iterates over the dataset (per column), and for each missing value in each column (time-serie), constructs the related prompt and also selects n number of examples randomly from other time series and feeds them into the selected LLM model.
   - `few_shot_prompting_rag_examples`: Few shot prompting at random method iterates over the dataset (per column), and for each missing value in each column (time-serie), constructs the related prompt and also selects n number of examples based on the relevance to the other time series using retrieval augmented generation and feeds them into the selected LLM model.
   - `few_shot_prompting_rag_contextual_info`: Few shot prompting at random method iterates over the dataset (per column), and for each missing value in each column (time-serie), constructs the related prompt and also selects n number of examples based on the relevance to the other time series using retrieval augmented generation, but instead of giving the whole sequences selected only the relevant information from the relevant time series and their nearest time points, constructs the "context" field, and feeds the prompt along with the context into the selected LLM model.
   - `hybrid_multiple_choice`: In hybrid multiple choice method, 3 different state of the art algorithms run at the beggining (SAITS, TimesNET and CSDI). Their results are collected and a prompt is constructed in the form of multiple choice based on the predictions of those algorithms. Then the final producted prompt is fed to the selected LLM in order to choose one of the available answers.
   - `hybrid_multiple_choice_v2`: In hybrid multiple choice v2 method, 3 different state of the art algorithms run at the beggining (SAITS, TimesNET and CSDI). Their results are collected and a prompt is constructed in the form of multiple choice based on the predictions of those algorithms. Then the final producted prompt is fed to the selected LLM in order to choose one of the available answers or the LLM do its own estimation.
 
 
### Secret parameters for llm-based methods
-  **`groq_api_key`** *(str, required)*   
  The groq api key of the user
 
-  **`api_token`** *(str, required)*    
  The groq token of the user
 

## Output
The output of TS-Impute has the following format:

```json
{
    "message": "TS-Impute Tool Executed Successfully",
	"output": {
		"imputed_timeseries": "path_of_imputed_file",
    },
	"metrics": {	
        "Mean absolute error": 4.073,
        "...": "additional metrics may be included"
    },
	"status": "success"
}
```
