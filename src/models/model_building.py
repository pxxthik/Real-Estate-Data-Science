import numpy as np
import pandas as pd

import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

import category_encoders as ce

import pickle

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="model_building.log")


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
        params = utils.load_params("params.yaml", "model_building", logger)

        # load data
        data_path = os.path.join("data", "processed")
        file_path = os.path.join(data_path, "train.csv")
        df = load_data(file_path)
        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        X = df.drop(columns=["price"])
        y = df["price"]

        # Applying log transform
        logger.info("Applying log transform")
        y_transformed = np.log1p(y)

        columns_to_encode = [
            "property_type",
            "sector",
            "balcony",
            "agePossession",
            "furnishing_type",
            "luxury_category",
            "floor_category",
        ]

        # Create the pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    [
                        "bedRoom",
                        "bathroom",
                        "built_up_area",
                        "servant room",
                        "store room",
                    ],
                ),
                ("cat", OrdinalEncoder(), columns_to_encode),
                (
                    "cat1",
                    OneHotEncoder(drop="first", sparse_output=False),
                    ["agePossession"],
                ),
                ("target_enc", ce.TargetEncoder(handle_unknown="ignore"), ["sector"]),
            ],
            remainder="passthrough",
        )

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_samples=params["max_samples"],
            max_features=params["max_features"],
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

        logger.info("Fitting the model")
        pipeline.fit(X, y_transformed)

        # Save the pipeline
        model_path = os.path.join("models")
        os.makedirs(model_path, exist_ok=True)

        logger.info("Saving the model")
        with open(os.path.join(model_path, "real_estate_predictor.pkl"), "wb") as file:
            pickle.dump(pipeline, file)
        
        logger.info("Saving artifact")
        with open(os.path.join(model_path, "df.pkl"), "wb") as file:
            pickle.dump(X, file)

    except Exception as e:
        logger.error(f"Error : {e}")
