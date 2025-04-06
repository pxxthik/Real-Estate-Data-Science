import numpy as np
import pandas as pd

import os

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="missing_value_imputation.log")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.debug("Loading Data")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

def impute_builtUpArea(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Imputing builtUpArea column")
    try:
        all_present_df = df.dropna(subset=['super_built_up_area', 'built_up_area', 'carpet_area'])

        super_to_built_up_ratio = (all_present_df['super_built_up_area'] / all_present_df['built_up_area']).median()
        carpet_to_built_up_ratio = (all_present_df['carpet_area'] / all_present_df['built_up_area']).median()

        # Case 1: both super_built_up_area and carpet_area present, but built_up_area is missing
        sbc_mask = (~df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (~df['carpet_area'].isnull())
        df.loc[sbc_mask, 'built_up_area'] = (
            ((df.loc[sbc_mask, 'super_built_up_area'] / super_to_built_up_ratio) +
             (df.loc[sbc_mask, 'carpet_area'] / carpet_to_built_up_ratio)) / 2
        ).round()

        # Case 2: super_built_up_area present, carpet_area and built_up_area missing
        sb_mask = (~df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())
        df.loc[sb_mask, 'built_up_area'] = (df.loc[sb_mask, 'super_built_up_area'] / super_to_built_up_ratio).round()

        # Case 3: carpet_area present, super_built_up_area and built_up_area missing
        c_mask = (df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (~df['carpet_area'].isnull())
        df.loc[c_mask, 'built_up_area'] = (df.loc[c_mask, 'carpet_area'] / carpet_to_built_up_ratio).round()

        # Handle anomaly: high price but low built_up_area
        anamoly_mask = (df['built_up_area'] < 2000) & (df['price'] > 2.5)
        df.loc[anamoly_mask, 'built_up_area'] = df.loc[anamoly_mask, 'area']

        df.drop(columns=['area', 'areaWithType', 'super_built_up_area', 'carpet_area', 'area_room_ratio'], inplace=True)

        return df
    except Exception as e:
        logger.error(f"Error imputing builtUpArea: {e}")
        return pd.DataFrame()

def mode_based_imputation(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


def mode_based_imputation2(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']

def mode_based_imputation3(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


def impute_AgePossion(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Imputing agePossession column")
    try:
        df['agePossession'] = df.apply(mode_based_imputation,axis=1)
        df['agePossession'] = df.apply(mode_based_imputation2,axis=1)
        df['agePossession'] = df.apply(mode_based_imputation3,axis=1)
        return df
    except Exception as e:
        logger.error(f"Error imputing agePossession: {e}")
        return pd.DataFrame()

def impute_floorNum(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Imputing floorNum column")
    try:
        median = df[df['property_type'] == 'house']['floorNum'].median()
        df['floorNum'] = df['floorNum'].fillna(median)
        return df
    except Exception as e:
        logger.error(f"Error imputing floorNum: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        data_path = os.path.join("data", "interim")
        file_path = os.path.join(data_path, "gurgaon_properties_outlier_treated.csv")

        df = load_data(file_path)

        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")
        
        # Pipeline
        df = (
            df.pipe(impute_builtUpArea)
            .pipe(impute_AgePossion)
            .pipe(impute_floorNum)
        )
        
        df.drop(columns=['facing'],inplace=True)
        df.drop(index=[2536],inplace=True)

        data_path = os.path.join("data", "interim")
        utils.save_data(df, data_path, "gurgaon_properties_missing_value_imputation.csv", logger=logger)
        

    except Exception as e:
        logger.error(f"Error loading data: {e}")
