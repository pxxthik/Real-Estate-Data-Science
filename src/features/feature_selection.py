import numpy as np
import pandas as pd

import os

from sklearn.preprocessing import OrdinalEncoder

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="feature_selection.log")


def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None  # or "Undefined" or any other label for scores outside the defined bins


def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None  # or "Undefined" or any other label for floors outside the defined bins


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Dropping unnecessary columns: ['society', 'price_per_sqft']")
    try:
        return df.drop(columns=["society", "price_per_sqft"])
    except Exception as e:
        logger.error(f"Error dropping columns: {e}")
        return pd.DataFrame()


def add_categorized_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Adding categorized columns: 'luxury_category', 'floor_category'")
    try:
        df["luxury_category"] = df["luxury_score"].apply(categorize_luxury)
        df["floor_category"] = df["floorNum"].apply(categorize_floor)
        return df.drop(columns=["floorNum", "luxury_score"])
    except Exception as e:
        logger.error(f"Error adding categorized columns: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        data_path = os.path.join("data", "interim")
        file_path = os.path.join(
            data_path, "gurgaon_properties_missing_value_imputation.csv"
        )
        df = utils.load_data(file_path, logger)
        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        # Pipeline
        df = df.pipe(drop_unnecessary_columns).pipe(add_categorized_columns)

        df = df.drop(columns=["pooja room", "study room", "others"])

        data_path = os.path.join("data", "processed")
        utils.save_data(
            df,
            data_path,
            "gurgaon_properties_post_feature_selection.csv",
            logger=logger,
        )

    except Exception as e:
        logger.error(f"Error loading data: {e}")
