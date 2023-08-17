from datetime import datetime, timedelta
from os.path import dirname, abspath
import os
import sys
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator, get_current_context

import pandas as pd
import logging
import json
from pathlib import Path
from mlflow.tracking import MlflowClient

ROOT = dirname(dirname(abspath(__file__)))

ARTIFACTS = ROOT+"/artifacts"
MLFLOW_TRACKING_URI = "http://localhost:5000"
PHASE_ID = "1"
PROB_NAME = "Customer_Churn"
EXPERIMENT_NAME = "v1"
MLFLOW_MODEL_PATH_PREFIX = "model"
RUN_INFO = ARTIFACTS + "/run_info.json"
EVALUATION_RESULT = ARTIFACTS + "/evaluation.json"



#=========================UTILS==============================
class RunInfo:
    def __init__(self, run_id) -> None:
        self.path = RUN_INFO
        self.run_id = run_id

    def save(self):
        run_info = {
            "run_id": self.run_id,
        }
        dump_json(run_info, self.path)

    @staticmethod
    def load(path):
        data = load_json(path)
        run_info = RunInfo(data["run_id"])
        return run_info

class Log:
    log: logging.Logger = None

    def __init__(self, name="") -> None:
        if Log.log == None:
            Log.log = self._init_logger(name)

    def _init_logger(self, name):
        logger = logging.getLogger(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.DEBUG)
        return logger
    
def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }

def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path
def inspect_dir(path):
    Log().log.info(f"inspect_dir {path}")
    path = Path(path)
    if not path.exists():
        Log().log.info(f"Path {path} doesn't exist")
        return
    elif path.is_file():
        Log().log.info(f"Path {path} is file")
        return

    paths = os.listdir(path)
    paths = sorted(paths)
    for path in paths:
        Log().log.info(path)


def inspect_curr_dir():
    cwd = os.getcwd()
    Log().log.info(f"current dir: {cwd}")
    inspect_dir(cwd)


def load_df(path) -> pd.DataFrame:
    Log().log.info(f"start load_df {path}")
    df = pd.read_parquet(path, engine="fastparquet")
    return df


def to_parquet(df: pd.DataFrame, path):
    Log().log.info(f"start to_parquet {path}")
    df.to_parquet(path, engine="fastparquet")


def dump_json(dict_obj: dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict_obj, f)


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


#==============================================================


#========================DEFINE-TASK=================================

def data_preparation(**kwargs):
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    
    data_path = os.path.join(ROOT, "data/data.csv")

    # data_url = dvc.api.get_url(path=data_path)

    data_df = pd.read_csv(data_path, nrows=5000)

    data_df.loc[data_df.SeniorCitizen==0,'SeniorCitizen'] = "No"   #convert 0 to No in all data instances
    data_df.loc[data_df.SeniorCitizen==1,'SeniorCitizen'] = "Yes"  #convert 1 to Yes in all data instances

    data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'],errors='coerce')
    #Fill the missing values with with the median value
    data_df['TotalCharges'] = data_df['TotalCharges'].fillna(data_df['TotalCharges'].median())


    data_df.drop(["customerID"],axis=1,inplace = True)

    # Encode categorical features

    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})

    ## Encoding target feature
    data_df['Churn'] = data_df[['Churn']].apply(binary_map)

    # Encoding gender category
    data_df['gender'] = data_df['gender'].map({'Male':1, 'Female':0})

    #Encoding other binary category
    binary_list = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    data_df[binary_list] = data_df[binary_list].apply(binary_map)

    #Encoding the other categoric features with more than two categories
    data_df = pd.get_dummies(data_df, drop_first=True)
    print(data_df.head())
    #feature scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    data_df['tenure'] = sc.fit_transform(data_df[['tenure']])
    data_df['MonthlyCharges'] = sc.fit_transform(data_df[['MonthlyCharges']])
    data_df['TotalCharges'] = sc.fit_transform(data_df[['TotalCharges']])

    X = data_df.drop('Churn', axis=1)
    y = data_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

    X_train.to_csv(os.path.join(ROOT, "data/X_train.csv"), index = False)
    X_test.to_csv(os.path.join(ROOT, "data/X_test.csv"), index = False)
    y_train.to_csv(os.path.join(ROOT, "data/y_train.csv"), index = False)
    y_test.to_csv(os.path.join(ROOT, "data/y_test.csv"), index= False)

    return True

def training_model():

    import mlflow
    from mlflow.models.signature import infer_signature
    import xgboost as xgb
    import numpy as np
    from sklearn.model_selection import train_test_split
    #Import metric for performance evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
    import pandas as pd
    import uuid

    #Setup tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()



    x_train= pd.read_csv(os.path.join(ROOT, "data/X_train.csv"))
    y_train = pd.read_csv(os.path.join(ROOT, "data/y_train.csv"))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    model_params = {'learning_rate': 0.06135832942699447,
                    'max_depth': 61,
                    'min_child_weight': 8.106374633926494,
                    'reg_alpha': 0.1371929596536868,
                    'reg_lambda': 0.0026689058379182856
                    }
    if len(np.unique(y_train)) == 2:
        objective = "binary:logistic"
    else:
        objective = "multi:softprob"
    print(y_train)
    print(len(y_train))
    model = xgb.XGBClassifier(objective=objective, **model_params)
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_val)
    auc_score = roc_auc_score(y_val, predictions, multi_class='ovr')
    acc_score = accuracy_score(y_val, predictions)
    pre_score = precision_score(y_val, predictions)
    rec_score = recall_score(y_val, predictions)
    f_score = f1_score(y_val, predictions, average='weighted')


    metrics = { "test_acc" : acc_score,
                "test_pre" : pre_score,
                "test_recall" : rec_score,
                "test_f1" : f_score,
                "test_auc": auc_score}


    # Log metadata
    mlflow.set_tag("mlflow.runName", str(uuid.uuid1())[:8])
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    signature = infer_signature(x_train, model.predict(x_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=MLFLOW_MODEL_PATH_PREFIX,
        signature=signature,
    )
    mlflow.end_run()

    # Inspect metadata
    run_id = mlflow.last_active_run().info.run_id

    for key, data in fetch_logged_data(run_id).items():
        Log().log.info("\n---------- logged {} ----------".format(key))
        Log().log.info(data)

    # Write latest run_id to file
    run_info = RunInfo(run_id)
    run_info.save()
    inspect_dir(run_info.path)
    return True

def evaluation_model():

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
    import mlflow

    run_info = RunInfo.load(RUN_INFO)
    model = mlflow.pyfunc.load_model(
        f"runs:/{run_info.run_id}/{MLFLOW_MODEL_PATH_PREFIX}"
    )
    x_test= pd.read_csv(os.path.join(ROOT, "data/X_test.csv"))
    y_test = pd.read_csv(os.path.join(ROOT, "data/y_test.csv"))

    predictions = model.predict(x_test)
    auc_score = roc_auc_score(y_test, predictions, multi_class='ovr')
    acc_score = accuracy_score(y_test, predictions)
    pre_score = precision_score(y_test, predictions)
    rec_score = recall_score(y_test, predictions)
    f_score = f1_score(y_test, predictions, average='weighted')


    metrics = { "test_acc" : acc_score,
                "test_pre" : pre_score,
                "test_recall" : rec_score,
                "test_f1" : f_score,
                "test_auc": auc_score}
    
    
    # Write evaluation result to file
    dump_json(metrics, EVALUATION_RESULT)
    Log().log.info(f"eval result: {metrics}")
    
#================================================================================================

with DAG(
    'customer_churn_classifier',
    description="Pipeline for training and deploying a classifier of toxic comments",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["tutorial"]
) as dag:


    data_preparation_task = BashOperator(
        task_id="data_preparation_task",
        python_callable=data_preparation
    )

    model_training_task = BashOperator(
        task_id="model_training_task",
        python_callable=training_model
    )

    model_evalution_task = BashOperator(
        task_id="model_evaluation_task",
        python_callable=evaluation_model    
    )
    (
    data_preparation_task
    >>model_training_task
    >>model_evalution_task)