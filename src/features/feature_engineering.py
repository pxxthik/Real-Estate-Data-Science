import numpy as np
import pandas as pd
import re
import os
import ast

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans

import src.features.helper as helper
import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="feature_engineering.log")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.debug("Loading Data")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


# This function extracts the Super Built up area
def get_super_built_up_area(text):
    match = re.search(r"Super Built up area (\d+\.?\d*)", text)
    if match:
        return float(match.group(1))
    return None


# This function extracts the Built Up area or Carpet area
def get_area(text, area_type):
    match = re.search(area_type + r"\s*:\s*(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))
    return None


# This function checks if the area is provided in sq.m. and converts it to sqft if needed
def convert_to_sqft(text, area_value):
    if area_value is None:
        return None
    match = re.search(r"{} \((\d+\.?\d*) sq.m.\)".format(area_value), text)
    if match:
        sq_m_value = float(match.group(1))
        return sq_m_value * 10.7639  # conversion factor from sq.m. to sqft
    return area_value


def convert_scale(row):
    if np.isnan(row["area"]) or np.isnan(row["built_up_area"]):
        return row["built_up_area"]
    else:
        if round(row["area"] / row["built_up_area"]) == 9.0:
            return row["built_up_area"] * 9
        elif round(row["area"] / row["built_up_area"]) == 11.0:
            return row["built_up_area"] * 10.7
        else:
            return row["built_up_area"]


# Function to extract plot area from 'areaWithType' column
def extract_plot_area(area_with_type):
    match = re.search(r"Plot area (\d+\.?\d*)", area_with_type)
    return float(match.group(1)) if match else None


def process_areaWithType(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    try:
        logger.debug("Processing areaWithType column")
        # Extract Super Built up area and convert to sqft if needed
        df["super_built_up_area"] = df["areaWithType"].apply(get_super_built_up_area)
        df["super_built_up_area"] = df.apply(
            lambda x: convert_to_sqft(x["areaWithType"], x["super_built_up_area"]),
            axis=1,
        )

        # Extract Built Up area and convert to sqft if needed
        df["built_up_area"] = df["areaWithType"].apply(
            lambda x: get_area(x, "Built Up area")
        )
        df["built_up_area"] = df.apply(
            lambda x: convert_to_sqft(x["areaWithType"], x["built_up_area"]), axis=1
        )

        # Extract Carpet area and convert to sqft if needed
        df["carpet_area"] = df["areaWithType"].apply(
            lambda x: get_area(x, "Carpet area")
        )
        df["carpet_area"] = df.apply(
            lambda x: convert_to_sqft(x["areaWithType"], x["carpet_area"]), axis=1
        )

        # Handle missing values
        logger.debug("Handling missing values")
        all_nan_df = df[
            (
                (df["super_built_up_area"].isnull())
                & (df["built_up_area"].isnull())
                & (df["carpet_area"].isnull())
            )
        ][
            [
                "price",
                "property_type",
                "area",
                "areaWithType",
                "super_built_up_area",
                "built_up_area",
                "carpet_area",
            ]
        ]
        all_nan_df["built_up_area"] = all_nan_df["areaWithType"].apply(
            extract_plot_area
        )
        all_nan_df["built_up_area"] = all_nan_df.apply(convert_scale, axis=1)
        df.update(all_nan_df)

        return df

    except Exception as e:
        logger.error(f"Error processing areaWithType: {e}")
        return pd.DataFrame()


def process_additionalRooms(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Processing additionalRooms column")
    df = data.copy()
    try:
        # List of new columns to be created
        new_cols = ["study room", "servant room", "store room", "pooja room", "others"]

        # Populate the new columns based on the "additionalRoom" column
        for col in new_cols:
            df[col] = df["additionalRoom"].str.contains(col).astype(int)
        return df

    except Exception as e:
        logger.error(f"Error processing additionalRooms: {e}")
        return pd.DataFrame()


def categorize_age_possession(value):
    if pd.isna(value):
        return "Undefined"
    if (
        "0 to 1 Year Old" in value
        or "Within 6 months" in value
        or "Within 3 months" in value
    ):
        return "New Property"
    if "1 to 5 Year Old" in value:
        return "Relatively New"
    if "5 to 10 Year Old" in value:
        return "Moderately Old"
    if "10+ Year Old" in value:
        return "Old Property"
    if "Under Construction" in value or "By" in value:
        return "Under Construction"
    try:
        # For entries like 'May 2024'
        int(value.split(" ")[-1])
        return "Under Construction"
    except:
        return "Undefined"


def process_agePosession(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Processing agePosession column")
    df = data.copy()
    try:
        df["agePossession"] = df["agePossession"].apply(categorize_age_possession)
        return df
    except Exception as e:
        logger.error(f"Error processing agePosession: {e}")
        return pd.DataFrame()


def process_furnishDetails(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Processing furnishDetails column")
    df = data.copy()
    try:
        # Extract all unique furnishings from the furnishDetails column
        all_furnishings = []
        for detail in df["furnishDetails"].dropna():
            furnishings = (
                detail.replace("[", "").replace("]", "").replace("'", "").split(", ")
            )
            all_furnishings.extend(furnishings)
        unique_furnishings = list(set(all_furnishings))

        # Define a function to extract the count of a furnishing from the furnishDetails
        def get_furnishing_count(details, furnishing):
            if isinstance(details, str):
                if f"No {furnishing}" in details:
                    return 0
                pattern = re.compile(f"(\d+) {furnishing}")
                match = pattern.search(details)
                if match:
                    return int(match.group(1))
                elif furnishing in details:
                    return 1
            return 0

        # Simplify the furnishings list by removing "No" prefix and numbers
        columns_to_include = [
            re.sub(r"No |\d+", "", furnishing).strip()
            for furnishing in unique_furnishings
        ]
        columns_to_include = list(set(columns_to_include))  # Get unique furnishings
        columns_to_include = [
            furnishing for furnishing in columns_to_include if furnishing
        ]  # Remove empty strings

        # Create new columns for each unique furnishing and populate with counts
        for furnishing in columns_to_include:
            df[furnishing] = df["furnishDetails"].apply(
                lambda x: get_furnishing_count(x, furnishing)
            )

        # Create the new dataframe with the required columns
        furnishings_df = df[["furnishDetails"] + columns_to_include]

        furnishings_df = furnishings_df.drop(columns=["furnishDetails"])

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(furnishings_df)

        n_clusters = 3
        # Fit the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_data)

        # Predict the cluster assignments for each row
        cluster_assignments = kmeans.predict(scaled_data)

        df = df.iloc[:, :-18]
        df["furnishing_type"] = cluster_assignments

        mapping = {
            0: 'unfurnished',
            1: 'semifurnished',
            2: 'furnished'
        }

        df['furnishing_type'] = df['furnishing_type'].map(mapping)

        return df
    except Exception as e:
        logger.error(f"Error processing furnishDetails: {e}")
        return pd.DataFrame()


def process_features(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Processing features column")
    df = data.copy()
    try:

        app_df = load_data(os.path.join("data", "raw", "appartments.csv"))
        app_df["PropertyName"] = app_df["PropertyName"].str.lower()

        temp_df = df[df["features"].isnull()]

        x = temp_df.merge(
            app_df, left_on="society", right_on="PropertyName", how="left"
        )["TopFacilities"]
        df.loc[temp_df.index, "features"] = x.values

        # Convert the string representation of lists in the 'features' column to actual lists
        df["features_list"] = df["features"].apply(
            lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith("[") else []
        )

        # Use MultiLabelBinarizer to convert the features list into a binary matrix
        mlb = MultiLabelBinarizer()
        features_binary_matrix = mlb.fit_transform(df["features_list"])

        # Convert the binary matrix into a DataFrame
        features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)

        # Define the weights for each feature as provided
        weights = helper.get_weights()
        # Calculate luxury score for each row
        luxury_score = (
            features_binary_df[list(weights.keys())]
            .multiply(list(weights.values()))
            .sum(axis=1)
        )
        df["luxury_score"] = luxury_score

        return df
    except Exception as e:
        logger.error(f"Error processing features: {e}")
        return pd.DataFrame()


if __name__ == "__main__":

    data_path = os.path.join("data", "interim")
    file_path = os.path.join(data_path, "gurgaon_properties_cleaned_v1.csv")

    df = load_data(file_path)
    if df.empty:
        raise ValueError("Data loading failed: Empty DataFrame")

    # Pipeline
    df = (
        df.pipe(process_areaWithType)
        .pipe(process_additionalRooms)
        .pipe(process_agePosession)
        .pipe(process_furnishDetails)
        .pipe(process_features)
    )

    # cols to drop -> nearbyLocations,furnishDetails, features,features_list, additionalRoom
    df.drop(
        columns=[
            "nearbyLocations",
            "furnishDetails",
            "features",
            "features_list",
            "additionalRoom",
        ],
        inplace=True,
    )

    data_path = os.path.join("data", "interim")
    utils.save_data(df, data_path, "gurgaon_properties_cleaned_v2.csv", logger=logger)
