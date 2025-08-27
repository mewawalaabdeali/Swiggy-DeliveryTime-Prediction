import pandas as pd
import yaml
import joblib
import logging 
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path

TARGET = 'time_taken'
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

def load_data(data_path:Path)-> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)

    except FileNotFoundError:
        logger.error("File not found ")
    return df

def read_params(file_path):
    with open(file_path, "r") as f:
        params_file = yaml.safe_load(f)
    return params_file
    
def save_model(model, save_dir:Path, model_name:str):
    #form the save location
    save_location = save_dir/model_name
    joblib.dump(value=model, filename=save_location)


def save_transformer(transformer, save_dir:Path, transformer_name:str):
    save_location = save_dir/transformer_name
    joblib.dump(transformer, save_location)

def train_model(model, X_train:pd.DataFrame, y_train):
    model.fit(X_train, y_train)
    return model

def make_X_and_y(data:pd.DataFrame, target_column:str):
    X = data.drop(columns=[target_column])
    y=data[target_column]
    return X,y

if __name__=="__main__":
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path/"data"/"processed"/"train_trans.csv"
    params_file_path = root_path/"params.yaml"

    training_data = load_data(data_path)
    logger.info("Training Data Read Succesfully")

    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info("Dataset splitting completed")

    model_params = read_params(params_file_path)['Train']

    rf_params = model_params['Random_Forest']
    logger.info("Random Forest Parameters Read")

    rf = RandomForestRegressor(**rf_params)
    logger.info("Built Random Forest Model")

    lgbm_params = model_params['LightGBM']
    logger.info("Light GBM parameters read succesful")
    lgbm = LGBMRegressor(**lgbm_params)
    logger.info("Build Light GBM Model")

    lr = LinearRegression()
    logger.info("Meta Model built")

    power_tranformer = PowerTransformer()
    logger.info("Target Transformer built")

    stacking_reg = StackingRegressor(estimators=[("rf_model", rf),
                                                 ("lgbm_model", lgbm)],
                                                 final_estimator=lr,
                                                 cv=5, n_jobs=1)
    logger.info("Stacking regressor built")

    model = TransformedTargetRegressor(regressor=stacking_reg, transformer=power_tranformer)
    logger.info("MOdels wrapped inside wrapper")
    train_model(model, X_train, y_train)
    logger.info("MOdel training completed")

    model_filename = "model.joblib"
    model_save_dir = root_path/"models"

    model_save_dir.mkdir(exist_ok=True)

    stacking_model = model.regressor_
    transformer = model.transformer_

    save_model(model=model, save_dir = model_save_dir, model_name=model_filename)
    logger.info("Trained model saved to location")

    stacking_filename = "stacking_regressor.joblib"
    save_model(model=stacking_model, save_dir=model_save_dir, model_name=stacking_filename)
    logger.info("Trained model saved to location")

    transformer_filename = "power_transformer.joblib"
    transformer_save_dir =model_save_dir
    save_transformer(transformer, transformer_save_dir, transformer_filename)
    logger.info("Transformer saved to location")