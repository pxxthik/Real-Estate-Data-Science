import numpy as np
import pandas as pd

import os

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="merge_flats_and_houses.log")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.debug("Loading data from %s", file_path)
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error("Error loading data: %s", e)
        return pd.DataFrame()
    
if __name__ == "__main__":
    try:
        data_path = os.path.join("data", "interim")
        flats_file_path = os.path.join(data_path, "flats.csv")
        houses_file_path = os.path.join(data_path, "houses.csv")
        flats_df = load_data(flats_file_path)
        houses_df = load_data(houses_file_path)
        if flats_df.empty or houses_df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        # Merge the dataframes
        merged_df = pd.concat([flats_df, houses_df], ignore_index=True)

        # Shuffle the merged dataframe
        merged_df = merged_df.sample(merged_df.shape[0],ignore_index=True)

        # Save the merged dataframe
        data_path = os.path.join("data", "interim")
        utils.save_data(merged_df, data_path, "gurgaon_properties.csv", logger=logger)

    except Exception as e:
        logger.error("Error merging data: %s", e)
