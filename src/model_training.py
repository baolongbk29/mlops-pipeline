import mlflow
from mlflow.models.signature import infer_signature
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
#Import metric for performance evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import pandas as pd
import uuid
from utils import *
from mlflow.tracking import MlflowClient


Log(AppConst.MODEL_TRAINING)
AppPath()

def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


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

def training_model():

    # Setup tracking server
    config = Config()
    Log().log.info(f"config: {config.__dict__}")
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    Log().log.info((mlflow.get_tracking_uri(), mlflow.get_artifact_uri()))
    mlflow.sklearn.autolog()



    x_train= pd.read_csv(AppPath.TRAIN_X)
    y_train = pd.read_csv(AppPath.TRAIN_Y)

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
        artifact_path=AppConst.MLFLOW_MODEL_PATH_PREFIX,
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

if __name__ == "__main__":
    training_model()