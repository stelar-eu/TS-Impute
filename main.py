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
            time_column = parameters.get("time_column", "time")
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
            df_missing = pd.read_csv(missing_local_path, header=header, sep=sep)

            if "ground_truth" in json_blob["input"]:
                gt_s3_path = json_blob["input"]["ground_truth"][0]
                gt_original_filename = os.path.basename(gt_s3_path)
                gt_local_path = os.path.join(os.getcwd(), gt_original_filename)
                mc.get_object(s3_path=gt_s3_path, local_path=gt_local_path)
                df_gt = pd.read_csv(gt_local_path, header=header, sep=sep)
            else:
                df_gt = None

            metrics = {}
            if tuning:
                if default:
                    hyper_params = tsi.default_hyperparameter_tuning_imputation_params()
                else:
                    hyper_params = parameters.get("params", tsi.default_hyperparameter_tuning_imputation_params())
                
                
                n_trials = parameters.get("n_trials", 20)

                best_params, logs = tsi.perform_hyperparameter_tuning(missing_df=df_missing,
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

            # output 
            out_obj = json_blob["output"]["imputed_timeseries"]
            output_file = f"imputed_{missing_original_filename}.csv"
            df_imputed.to_csv(output_file, index=False)
            mc.put_object(file_path=output_file, s3_path=out_obj)

            if df_gt is not None:
                df_missing.set_index(time_column, inplace=True)

                df_gt.set_index(time_column, inplace=True)

                df_imputed.set_index(time_column, inplace=True)

                calc_metrics = tsi.compute_metrics(df_gt, df_missing, df_imputed)
                metrics[f"Imputation results for {imp_method}"] = calc_metrics

            return {
                "message": "Tool Executed Succesfully",
                "output": {"imputed_timeseries": out_obj},
                "metrics": metrics,
                "status": "success",
            }
        else:
            # code for llm methods
            return {
                "message": "Tool Executed Succesfully",
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
