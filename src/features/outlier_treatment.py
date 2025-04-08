import numpy as np
import pandas as pd

import os

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="outlier_treatment.log")


def treat_pricePerSqft(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Treat pricePerSqft column")
    df = data.copy()
    try:

        # Calculate the IQR for the 'price' column
        Q1 = df["price_per_sqft"].quantile(0.25)
        Q3 = df["price_per_sqft"].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers_sqft = df[
            (df["price_per_sqft"] < lower_bound) | (df["price_per_sqft"] > upper_bound)
        ]
        outliers_sqft = outliers_sqft.copy()
        outliers_sqft["area"] = outliers_sqft["area"].apply(
            lambda x: x * 9 if x < 1000 else x
        )
        outliers_sqft["price_per_sqft"] = round(
            (outliers_sqft["price"] * 10000000) / outliers_sqft["area"]
        )
        df.update(outliers_sqft)

        df = df[df["price_per_sqft"] <= 50000]

        return df
    except Exception as e:
        logger.error(f"Error treating pricePerSqft: {e}")
        return pd.DataFrame()


def treat_area(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Treat area column")
    df = data.copy()
    try:
        df = df[df["area"] < 100000]
        df = df.drop(index=[818, 1796, 1123, 2, 2356, 2503, 1471])

        df.loc[48, "area"] = 115 * 9
        df.loc[300, "area"] = 7250
        df.loc[2666, "area"] = 5800
        df.loc[1358, "area"] = 2660
        df.loc[3195, "area"] = 2850
        df.loc[2131, "area"] = 1812
        df.loc[3088, "area"] = 2160
        df.loc[3444, "area"] = 1175

        return df
    except Exception as e:
        logger.error(f"Error treating area: {e}")
        return pd.DataFrame()


def treat_bedreoom(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Treat bedreoom column")
    df = data.copy()
    try:
        df = df[df["bedRoom"] <= 10]
        return df
    except Exception as e:
        logger.error(f"Error treating bedreoom: {e}")
        return pd.DataFrame()


def treat_carpetArea(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Treat carpetArea column")
    df = data.copy()
    try:
        df.loc[2131, "carpet_area"] = 1812
        return df
    except Exception as e:
        logger.error(f"Error treating carpetArea: {e}")
        return pd.DataFrame()


def reTreat_pricePerSqft(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Treat pricePerSqft column")
    df = data.copy()
    try:
        df["price_per_sqft"] = round((df["price"] * 10000000) / df["area"])
        return df
    except Exception as e:
        logger.error(f"Error treating pricePerSqft: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        data_path = os.path.join("data", "interim")
        file_path = os.path.join(data_path, "gurgaon_properties_cleaned_v2.csv")

        df = utils.load_data(file_path, logger).drop_duplicates()

        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        # Pipeline
        df = (
            df.pipe(treat_pricePerSqft)
            .pipe(treat_area)
            .pipe(treat_bedreoom)
            .pipe(treat_carpetArea)
            .pipe(reTreat_pricePerSqft)
        )

        df["area_room_ratio"] = df["area"] / df["bedRoom"]

        data_path = os.path.join("data", "interim")
        utils.save_data(
            df, data_path, "gurgaon_properties_outlier_treated.csv", logger=logger
        )

    except Exception as e:
        logger.error(f"Error loading data: {e}")
