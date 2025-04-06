import numpy as np
import pandas as pd

import os
import pickle

from sklearn.metrics import mean_absolute_error, r2_score
from dvclive import Live

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="model_evaluation.log")

def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.debug("Loading Data")
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        # loading params
        params = utils.load_params("params.yaml", "all", logger)

        # load data
        data_path = os.path.join("data", "processed")
        file_path = os.path.join(data_path, "test.csv")
        df = load_data(file_path)
        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        X_test = df.drop("price", axis=1)
        y_test = df["price"]

        # load model
        logger.info("Loading model")
        model_path = os.path.join("models")
        os.makedirs(model_path, exist_ok=True)

        with open(os.path.join(model_path, "real_estate_predictor.pkl"), "rb") as f:
            model = pickle.load(f)

        logger.info("Model loaded successfully")

        logger.info("Evaluating model")
        y_pred = np.expm1(model.predict(X_test))

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info("Recording metrics and params")
        with Live(save_dvc_exp=True) as live:
            live.log_metric("mae", mae)
            live.log_metric("r2", r2)

            for module, content in params.items():
                for key, value in content.items():
                    live.log_param(f"{module}.{key}", value)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
