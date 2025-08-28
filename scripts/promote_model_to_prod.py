import mlflow
import dagshub
import json
from mlflow import MlflowClient

import mlflow.client
dagshub.init(repo_owner='mewawalaabdeali',
            repo_name='Swiggy-DeliveryTime-Prediction',
            mlflow=True)

#set the tracking server
mlflow.set_tracking_uri('https://dagshub.com/mewawalaabdeali/Swiggy-DeliveryTime-Prediction.mlflow')

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

#get model name
model_name = load_model_information("run_information.json")['model_name']
stage = "Staging"

#get the latest version from staging 
client = MlflowClient()

#get the latest version of model in staging
latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
latest_model_version_staging = latest_versions[0].version

promotion_stage ="Production"

client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)