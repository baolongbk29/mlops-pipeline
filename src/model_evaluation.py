from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import mlflow
from utils import *

Log(AppConst.MODEL_EVALUATION)
AppPath()




def evaluation_model():

    run_info = RunInfo.load(AppPath.RUN_INFO)
    model = mlflow.pyfunc.load_model(
        f"runs:/{run_info.run_id}/{AppConst.MLFLOW_MODEL_PATH_PREFIX}"
    )
    x_test= pd.read_csv(AppPath.TEST_X)
    y_test = pd.read_csv(AppPath.TEST_Y)

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
    eval_result = EvaluationResult(acc_score, pre_score, rec_score, f_score, acc_score)
    dump_json(metrics, AppPath.EVALUATION_RESULT)
    Log().log.info(f"eval result: {eval_result}")
    eval_result.save()
    inspect_dir(eval_result.path)

if __name__ == "__main__":
    evaluation_model()