import numpy as np
import pandas as pd

import re
import os
import logging
import json

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src import utils as utils


# Function to parse and extract the required features from the PriceDetails column
def refined_parse_modified_v2(detail_str):
    try:
        details = json.loads(detail_str.replace("'", "\""))
    except:
        return {}

    extracted = {}
    for bhk, detail in details.items():
        # Extract building type
        extracted[f'building type_{bhk}'] = detail.get('building_type')

        # Parsing area details
        area = detail.get('area', '')
        area_parts = area.split('-')
        if len(area_parts) == 1:
            try:
                value = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area low {bhk}'] = value
                extracted[f'area high {bhk}'] = value
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None
        elif len(area_parts) == 2:
            try:
                extracted[f'area low {bhk}'] = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area high {bhk}'] = float(area_parts[1].replace(',', '').replace(' sq.ft.', '').strip())
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None

        # Parsing price details
        price_range = detail.get('price-range', '')
        price_parts = price_range.split('-')
        if len(price_parts) == 2:
            try:
                extracted[f'price low {bhk}'] = float(price_parts[0].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                extracted[f'price high {bhk}'] = float(price_parts[1].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                if 'L' in price_parts[0]:
                    extracted[f'price low {bhk}'] /= 100
                if 'L' in price_parts[1]:
                    extracted[f'price high {bhk}'] /= 100
            except:
                extracted[f'price low {bhk}'] = None
                extracted[f'price high {bhk}'] = None

    return extracted


def get_cosine_sim2(logger: logging):
    # load appartments
    data_path = os.path.join("data", "raw")
    file_path = os.path.join(data_path, "appartments.csv")
    df_appartments = utils.load_data(file_path, logger).drop(22)

    # Apply the refined parsing and generate the new DataFrame structure
    data_refined = []

    for _, row in df_appartments.iterrows():
        features = refined_parse_modified_v2(row['PriceDetails'])
        
        # Construct a new row for the transformed dataframe
        new_row = {'PropertyName': row['PropertyName']}
        
        # Populate the new row with extracted features
        for config in ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK', '6 BHK', '1 RK', 'Land']:
            new_row[f'building type_{config}'] = features.get(f'building type_{config}')
            new_row[f'area low {config}'] = features.get(f'area low {config}')
            new_row[f'area high {config}'] = features.get(f'area high {config}')
            new_row[f'price low {config}'] = features.get(f'price low {config}')
            new_row[f'price high {config}'] = features.get(f'price high {config}')
        
        data_refined.append(new_row)

    df_final_refined_v2 = pd.DataFrame(data_refined).set_index('PropertyName')

    df_final_refined_v2['building type_Land'] = df_final_refined_v2['building type_Land'].replace({'':'Land'})

    categorical_columns = df_final_refined_v2.select_dtypes(include=['object']).columns.tolist()
    ohe_df = pd.get_dummies(df_final_refined_v2, columns=categorical_columns, drop_first=True)
    ohe_df.fillna(0,inplace=True)

    # Initialize the scaler
    scaler = StandardScaler()

    # Apply the scaler to the entire dataframe
    ohe_df_normalized = pd.DataFrame(scaler.fit_transform(ohe_df), columns=ohe_df.columns, index=ohe_df.index)

    # Compute the cosine similarity matrix
    cosine_sim2 = cosine_similarity(ohe_df_normalized)

    return cosine_sim2
