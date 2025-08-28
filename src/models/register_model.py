import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging

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

#initialize dagshub
import dagshub
import mlflow.client
dagshub.init(repo_owner='mewawalaabdeali',
            repo_name='Swiggy-DeliveryTime-Prediction',
            mlflow=True)

#set the tracking server
mlflow.set_tracking_uri('https://dagshub.com/mewawalaabdeali/Swiggy-DeliveryTime-Prediction.mlflow')

#mlflow.set_experiment("DVC Pipeline")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)

    return run_info

if __name__=="__main__":
    root_path = Path(__file__).parent.parent.parent

    #run information file path
    run_info_path = root_path/"run_information.json"

    #register the model
    run_info = load_model_information(run_info_path)

    #get the run id
    run_id = run_info['run_id']
    model_name = run_info['model_name']

    #model to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"

    model_version = mlflow.register_model(model_uri=model_registry_path, name = model_name)

    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")

    #update the stage of the model to staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name = registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )

    logger.info("Model pushed to Staging.")
