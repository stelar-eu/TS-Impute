import json
import sys
import os
import traceback
from utils.mclient import MinioClient
import stelarImputation as tsi
import pandas as pd


def run(json_blob):
    try:

        minio = json_blob["minio"]

        mc = MinioClient(
            minio["endpoint_url"],
            minio["id"],
            minio["key"],
            secure=True,
            session_token=minio["skey"],
        )

        # check imp_method
        traditional_algorithms = ["iterativesvd", "rosl", "cdmissingvaluerecovery",
                                  "ogdimpute", "nmfmissingvaluerecovery", "dynammo", "grouse", "softimpute",
                                  "pca_mme", "svt", "spirit", "tkcm", "linearimpute", "meanimpute", "zeroimpute",
                                  "naive", "saits", "brits", "csdi", "usgan", "mrnn", "gpvae", "timesnet",
                                  "nonstationary_transformer", "autoformer"]

        # Common field
        parameters = json_blob["parameters"]
        imp_method = parameters.get("imp_method", "naive")
        
        if imp_method.lower() in traditional_algorithms:

            # parameters
            # Get column name and format
            time_column = parameters.get("time_column", "time")
            time_format = parameters.get("time_format", "%Y-%m-%d %H:%M:%S")
            tuning = parameters.get("hyperparameter_tuning", False)
            sep = parameters.get("sep", ",")
            header = parameters.get("header", 0)
            is_multivariate = parameters.get("is_multivariate", False)
            default = parameters.get("default", True)

            # Download the file from S3
            missing_s3_path = json_blob["input"]["missing"][0]
            missing_original_filename = os.path.basename(missing_s3_path)
            missing_local_path = os.path.join(os.getcwd(), missing_original_filename)
            mc.get_object(s3_path=missing_s3_path, local_path=missing_local_path)
            # Normalize header argument
            if header is True:
                header = 0
            elif header is False:
                header = None
            elif isinstance(header, (int, list, type(None))):
                # Valid Pandas options, pass through
                pass
            else:
                raise ValueError(f"Invalid header value: {header}")
            if missing_local_path.endswith('.csv'):
                df_missing = pd.read_csv(missing_local_path, header=header, sep=sep)
            elif missing_local_path.endswith(('.xlsx', '.xls')):
                df_missing = pd.read_excel(missing_local_path, header=header)
            else:
                raise ValueError("Unsupported file type")
            
            # columns with timeseries with no values at all in the missing data
            cols_to_drop = df_missing.columns[df_missing.isna().all()]

            # drop timeseries with no values 
            df_missing.drop(columns=cols_to_drop, inplace=True)

            original_order = df_missing[time_column].tolist()
            # Sort missing dataframe by datetime column (time_column) using temporary conversion
            df_missing = df_missing.iloc[pd.to_datetime(df_missing[time_column], format=time_format).argsort()]

            if "ground_truth" in json_blob["input"]:
                gt_s3_path = json_blob["input"]["ground_truth"][0]
                gt_original_filename = os.path.basename(gt_s3_path)
                gt_local_path = os.path.join(os.getcwd(), gt_original_filename)
                mc.get_object(s3_path=gt_s3_path, local_path=gt_local_path)
                if gt_local_path.endswith('.csv'):
                    df_gt = pd.read_csv(gt_local_path, header=header, sep=sep)
                elif gt_local_path.endswith(('.xlsx', '.xls')):
                    df_gt = pd.read_excel(gt_local_path, header=header)
                else:
                    raise ValueError("Unsupported file type")
                
                # drop timeseries with no values using the columns from the missing data
                df_gt.drop(columns=cols_to_drop, inplace=True)

                # Sort missing dataframe by datetime column (time_column) using temporary conversion
                df_gt = df_gt.iloc[pd.to_datetime(df_gt[time_column], format=time_format).argsort()]
            else:
                df_gt = None

            metrics = {}
            if tuning:
                if default:
                    hyper_params = tsi.default_hyperparameter_tuning_imputation_params()
                else:
                    hyper_params = parameters.get("params", tsi.default_hyperparameter_tuning_imputation_params())
                
                
                n_trials = parameters.get("n_trials", 20)
                tuning_top_k = parameters.get("tuning_top_k_fewer_missing", 1000) + 1
                
                if tuning_top_k >= (df_missing.shape[1] - 1):
                    hpt_df_missing = df_missing.copy()
                else:
                    nan_counts =df_missing.isnull().sum()
                    least_nan_cols = nan_counts.nsmallest(int(tuning_top_k)).index
                    hpt_df_missing = df_missing[least_nan_cols].copy()

                best_params, logs = tsi.perform_hyperparameter_tuning(missing_df=hpt_df_missing,
                                                                      algorithm=imp_method,
                                                                      params=hyper_params,
                                                                      n_trials=n_trials,
                                                                      time_column=time_column,
                                                                      is_multivariate=is_multivariate,
                                                                      preprocessing=False,
                                                                      index=False)

                params = best_params
                metrics["Hyperparameter tuning results"] = logs
            else:
                if default:
                    params = tsi.default_imputation_params()
                else:
                    params = parameters.get("params", tsi.default_imputation_params())

            df_imputed = tsi.run_imputation(missing=df_missing,
                                            algorithms=[imp_method],
                                            params=params,
                                            time_column=time_column,
                                            is_multivariate=is_multivariate,
                                            preprocessing=False,
                                            index=False)[imp_method]
            
            if df_gt is not None:
                df_missing.set_index(time_column, inplace=True)

                df_gt.set_index(time_column, inplace=True)

                df_imputed.set_index(time_column, inplace=True)

                calc_metrics = tsi.compute_metrics(df_gt, df_missing, df_imputed)
                metrics[f"Imputation results for {imp_method}"] = calc_metrics

                df_imputed.reset_index(inplace=True, names=[time_column])
                df_imputed.columns.name = ''
            
            positions = [df_imputed[time_column].tolist().index(val) for val in original_order]
            df_imputed = df_imputed.iloc[positions].reset_index(drop=True)

            # output 
            out_obj = json_blob["output"]["imputed_timeseries"]
            output_file = f"imputed_{missing_original_filename}"
            write_header = (header is not None) and (header is not False)
            if missing_original_filename.lower().endswith(".csv"):
                df_imputed.to_csv(output_file, index=False, header=write_header, sep=sep)
            elif missing_original_filename.lower().endswith((".xlsx", ".xls")):
                df_imputed.to_excel(output_file, index=False, header=write_header)
            else:
                raise ValueError(f"Unsupported file type in: {missing_original_filename}")

            mc.put_object(file_path=output_file, s3_path=out_obj)

            return {
                "message": "Tool Executed Successfully",
                "output": {"imputed_timeseries": out_obj},
                "metrics": metrics,
                "status": "success",
            }
        else:
            # code for llm methods
            return {
                "message": "Tool Executed Successfully",
                "output": {"imputed_timeseries": ""},
                "metrics": {},
                "status": "success",
            }
    except Exception:
        print(traceback.format_exc())
        return {
            "message": "An error occurred during data processing.",
            "error": traceback.format_exc(),
            "status": 500
        }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], "w") as o:
        o.write(json.dumps(response, indent=4))
