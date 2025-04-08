import numpy as np
import pandas as pd

import os
import pickle

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="data_viz.log")

def extract_latlong(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Extracting latlong")
        df['latitude'] = df['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')
        df['longitude'] = df['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')
        return df
    except Exception as e:
        logger.error(f"Error extracting latlong: {e}")
        return pd.DataFrame()

def merge_latlong(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Merging latlong")
        # load latlong
        data_path = os.path.join("data", "raw")
        file_path = os.path.join(data_path, "latlong.csv")
        latlong = utils.load_data(file_path, logger)

        # extract latlong
        latlong = extract_latlong(latlong)

        return df.merge(latlong, on='sector')
    
    except Exception as e:
        logger.error(f"Error merging latlong: {e}")
        return pd.DataFrame()



if __name__ == "__main__":
    try:
        # load data
        data_path = os.path.join("data", "interim")
        file_path = os.path.join(data_path, "gurgaon_properties_missing_value_imputation.csv")
        df = utils.load_data(file_path, logger)
        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")
        
        # merge latlong
        merged = merge_latlong(df)

        # Save
        data_path = os.path.join("models")
        utils.save_data(merged, data_path, "data_viz1.csv", logger=logger)

        # Load gurgaon properties
        data_path = os.path.join("data", "interim")
        file_path = os.path.join(data_path, "gurgaon_properties.csv")
        df1 = utils.load_data(file_path, logger)
        if df1.empty:
            raise ValueError("Data loading failed: Empty DataFrame")
        
        wordcloud_df = df1.merge(df, left_index=True, right_index=True)[['features','sector']]

        # Export wordcloud_df
        path = os.path.join("models")
        
        logger.info("Exporting wordcloud df")
        with open(os.path.join(path, "wordcloud_df.pkl"), "wb") as file:
            pickle.dump(wordcloud_df, file)

        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
