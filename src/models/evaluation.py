import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json

import dagshub
dagshub.init(repo_owner='mewawalaabdeali', repo_name='Swiggy-DeliveryTime-Prediction', mlflow=True)
#set the tracking server
mlflow.set_tracking_uri('https://dagshub.com/mewawalaabdeali/Swiggy-DeliveryTime-Prediction.mlflow')

mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logging.debug("Data Read succesfully")
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    except Exception as e:
        logger.error("Unknown exception %s",e)

    return df

def make_X_and_y(data:pd.DataFrame, target_column:str):
    X=data.drop(columns=[target_column])
    y=data[target_column]
    return X,y

def load_model(model_path:Path):
    model = joblib.load(model_path)
    return model

def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path, "w") as f:
        json.dump(info_dict, f, indent=4)


if __name__=="__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data load path
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    # model path
    model_path = root_path / "models" / "model.joblib"
    
    
    # load the training data
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")
    # load the test data
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")
    
    # split the train and test data
    X_train, y_train = make_X_and_y(train_data,TARGET)
    X_test, y_test = make_X_and_y(test_data,TARGET)
    logger.info("Data split completed")
    
    # load the model
    model = load_model(model_path)
    logger.info("Model Loaded successfully")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Prediction on data complete")

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    logger.info("error calculated")

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logger.info("r2 score calculated")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=1)

    logger.info("cross validation completed")

    mean_cv_score = -(cv_scores.mean())

    with mlflow.start_run() as run:
        mlflow.set_tag("model", "Swiggy Delivery Time Regressor")
        mlflow.log_params(model.get_params())

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae",test_mae)
        mlflow.log_metric("train_r2",train_r2)
        mlflow.log_metric("test_r2",test_r2)
        mlflow.log_metric("mean_cv_score",-(cv_scores.mean()))

        mlflow.log_metrics({f"CV {num}": score for num,score in enumerate(-cv_scores)})

        train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data, targets = TARGET)

        mlflow.log_input(dataset=train_data_input, context="training")
        mlflow.log_input(dataset=test_data_input, context="validation")

        model_signature = mlflow.models.infer_signature(model_input=X_train.sample(20, random_state=42),
                                                        model_output = model.predict(X_train.sample(20, random_state=42)))
        
        mlflow.sklearn.log_model(model, "Swiggy-Deliver-TimePrediction", signature=model_signature)

        mlflow.log_artifact(root_path/"models"/"stacking_regressor.joblib")
        mlflow.log_artifact(root_path/"models"/"power_transformer.joblib")
        mlflow.log_artifact(root_path/"models"/"preprocessor.joblib")

        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Mlflow Logging complete and model logged")

        run_id = run.info.run_id
        model_name = "Swiggy-Deliver-TimePrediction"

        save_json_path = root_path/"run_information.json"
        save_model_info(save_json_path=save_json_path,
                        run_id=run_id,
                        artifact_path=artifact_uri,
                        model_name=model_name)
        logger.info("Model Information Saved")
                          

