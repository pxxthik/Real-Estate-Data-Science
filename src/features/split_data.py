import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import os
import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="splitting_data.log")


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
        params = utils.load_params("params.yaml", "split_data", logger)

        # load data
        data_path = os.path.join("data", "processed")
        file_path = os.path.join(
            data_path, "gurgaon_properties_post_feature_selection.csv"
        )
        df = load_data(file_path)
        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        # split data into train and test
        train, test = train_test_split(
            df, test_size=params["test_size"], random_state=params["random_state"]
        )

        # save train and test data
        data_path = os.path.join("data", "processed")
        utils.save_data(train, data_path, "train.csv", logger=logger)
        utils.save_data(test, data_path, "test.csv", logger=logger)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
