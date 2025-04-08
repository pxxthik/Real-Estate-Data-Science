import numpy as np
import pandas as pd

import os
import re
import logging

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from difflib import SequenceMatcher
import src.utils as utils

def extract_list(s):
    return re.findall(r"'(.*?)'", s)

def distance_to_meters(distance_str):
    try:
        if 'Km' in distance_str or 'KM' in distance_str:
            return float(distance_str.split()[0]) * 1000
        elif 'Meter' in distance_str or 'meter' in distance_str:
            return float(distance_str.split()[0])
        else:
            return None
    except:
        return None

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to group similar phrases
def group_similar_phrases(phrases):
    groups = {}
    for phrase in phrases:
        added = False
        for key in groups.keys():
            if similar(phrase, key) > 0.7:
                groups[key].append(phrase)
                added = True
                break
        if not added:
            groups[phrase] = [phrase]
    return groups


def get_location_df(logger: logging):
    # load appartments
    data_path = os.path.join("data", "raw")
    file_path = os.path.join(data_path, "appartments.csv")

    df = utils.load_data(file_path, logger).drop(22)


    df['TopFacilities'] = df['TopFacilities'].apply(extract_list)
    df['FacilitiesStr'] = df['TopFacilities'].apply(' '.join)

    all_locations = []

    for loc in df['LocationAdvantages'].dropna().apply(lambda x: eval(x).keys()):
        all_locations.extend(loc)

    all_locations = list(set(all_locations))

    # Group similar phrases
    groups = group_similar_phrases(all_locations)

    # Create a dictionary
    result_dict = {key: value for key, value in groups.items()}

    res = {}
    for key, values in result_dict.items():
        for value in values:
            res[value] = key
    
    def foo(d):
        d = eval(d)
        new_d = {}
        for key, value in d.items():
            new_d[res[key]] = value
        return new_d
    
    df['LocationAdvantages'] = df['LocationAdvantages'].apply(foo)

    # Extract distances for each location
    location_matrix = {}
    for index, row in df.iterrows():
        distances = {}
        for location, distance in row['LocationAdvantages'].items():
            distances[location] = distance_to_meters(distance)
        location_matrix[index] = distances

    # Convert the dictionary to a dataframe
    location_df = pd.DataFrame.from_dict(location_matrix, orient='index')
    location_df.index = df.PropertyName

    return location_df.fillna(54000)


def get_cosine_sim3(logger: logging):
    location_df = get_location_df(logger)

    # Initialize the scaler
    scaler = StandardScaler()

    # Apply the scaler to the entire dataframe
    location_df_normalized = pd.DataFrame(scaler.fit_transform(location_df), columns=location_df.columns, index=location_df.index)

    cosine_sim3 = cosine_similarity(location_df_normalized)

    return location_df, cosine_sim3
