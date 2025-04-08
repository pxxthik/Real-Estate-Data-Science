import numpy as np
import pandas as pd
import os
import re

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="preprocessing_houses.log")


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Dropping duplicates")
        df = df.drop_duplicates()
        return df
    except Exception as e:
        logger.error("Error dropping duplicates: %s", e)
        return pd.DataFrame()


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Dropping columns")
        columns_to_drop = ["link", "property_id"]
        df.drop(columns=columns_to_drop, inplace=True)
        return df
    except Exception as e:
        logger.error("Error dropping columns: %s", e)
        return pd.DataFrame()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Renaming columns")
        df.rename(columns={"rate": "price_per_sqft"}, inplace=True)
        return df
    except Exception as e:
        logger.error("Error renaming columns: %s", e)
        return pd.DataFrame()


def clean_society(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Cleaning society column")
        df["society"] = (
            df["society"]
            .apply(lambda name: re.sub(r"\d+(\.\d+)?\s?★", "", str(name)).strip())
            .str.lower()
        )
        df["society"] = df["society"].str.replace("nan", "independent")
        return df
    except Exception as e:
        logger.error("Error cleaning society column: %s", e)
        return pd.DataFrame()


def process_price(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Processing price column")
    try:
        logger.debug("Filtering out 'Price on Request'")
        df = df[df["price"] != "Price on Request"].copy()

        def treat_price(x):
            if type(x) == float:
                return x
            else:
                if x[1] == "Lac":
                    return round(float(x[0]) / 100, 2)
                else:
                    return round(float(x[0]), 2)

        logger.debug("Extracting price values")
        df["price"] = df["price"].str.split(" ").apply(treat_price)

        return df
    except Exception as e:
        logger.error("Error processing price column: %s", e)
        return pd.DataFrame()


def process_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Processing price_per_sqft column")
        df["price_per_sqft"] = (
            df["price_per_sqft"]
            .str.split("/")
            .str.get(0)
            .str.replace("₹", "")
            .str.replace(",", "")
            .str.strip()
            .astype("float")
        )
        return df
    except Exception as e:
        logger.error("Error processing price_per_sqft: %s", e)
        return pd.DataFrame()


def remove_bedroom_nulls(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Removing nulls in bedRoom")
        df = df[~df["bedRoom"].isnull()]
        return df
    except Exception as e:
        logger.error("Error removing bedroom nulls: %s", e)
        return pd.DataFrame()


def convert_bedroom(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Converting bedRoom to int")
        df = df.copy()
        df["bedRoom"] = df["bedRoom"].str.split(" ").str.get(0).astype("int")
        return df
    except Exception as e:
        logger.error("Error converting bedRoom: %s", e)
        return pd.DataFrame()


def process_bedroom(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_bedroom_nulls(df)
    df = convert_bedroom(df)
    return df


def convert_bathroom(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Converting bathroom to int")
        df["bathroom"] = df["bathroom"].str.split(" ").str.get(0).astype("int")
        return df
    except Exception as e:
        logger.error("Error converting bathroom: %s", e)
        return pd.DataFrame()


def process_balcony(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Processing balcony column")
        df["balcony"] = df["balcony"].str.split(" ").str.get(0).str.replace("No", "0")
        return df
    except Exception as e:
        logger.error("Error processing balcony: %s", e)
        return pd.DataFrame()


def handle_additional_room(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Handling additionalRoom column")
        df = df.copy()

        df["additionalRoom"] = df["additionalRoom"].fillna("not available")
        df["additionalRoom"] = df["additionalRoom"].str.lower()

        return df
    except Exception as e:
        logger.error("Error handling additionalRoom: %s", e)
        return pd.DataFrame()


def process_floor(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Processing floor column")
        df["noOfFloor"] = df["noOfFloor"].str.split(" ").str.get(0)
        df.rename(columns={"noOfFloor": "floorNum"}, inplace=True)
        return df
    except Exception as e:
        logger.error("Error processing floor: %s", e)
        return pd.DataFrame()


def handle_facing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Handling facing column")
        df["facing"] = df["facing"].fillna("NA")
        return df
    except Exception as e:
        logger.error("Error handling facing: %s", e)
        return pd.DataFrame()


def calculate_area(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Calculating area column")
        df["area"] = round((df["price"] * 10000000) / df["price_per_sqft"])
        return df
    except Exception as e:
        logger.error("Error calculating area: %s", e)
        return pd.DataFrame()


def add_property_type(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Adding property_type column")
        df.insert(loc=1, column="property_type", value="house")
        return df
    except Exception as e:
        logger.error("Error adding property_type: %s", e)
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        data_path = os.path.join("data", "raw")
        file_path = os.path.join(data_path, "houses.csv")
        df = utils.load_data(file_path, logger)
        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        # Pipeline
        df = (
            df.pipe(drop_duplicates)
            .pipe(drop_columns)
            .pipe(rename_columns)
            .pipe(clean_society)
            .pipe(process_price)
            .pipe(process_price_per_sqft)
            .pipe(process_bedroom)
            .pipe(convert_bathroom)
            .pipe(process_balcony)
            .pipe(handle_additional_room)
            .pipe(process_floor)
            .pipe(handle_facing)
            .pipe(calculate_area)
            .pipe(add_property_type)
        )

        data_path = os.path.join("data", "interim")
        utils.save_data(df, data_path, "houses.csv", logger=logger)

    except Exception as main_e:
        logger.error("Pipeline failed: %s", main_e)
