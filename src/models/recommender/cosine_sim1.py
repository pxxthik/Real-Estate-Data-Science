import numpy as np
import pandas as pd

import re
import os
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import utils as utils

def extract_list(s):
    return re.findall(r"'(.*?)'", s)

def get_cosine_sim1(logger: logging):
    # load appartments
    data_path = os.path.join("data", "raw")
    file_path = os.path.join(data_path, "appartments.csv")

    df = utils.load_data(file_path, logger).drop(22)


    df['TopFacilities'] = df['TopFacilities'].apply(extract_list)
    df['FacilitiesStr'] = df['TopFacilities'].apply(' '.join)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['FacilitiesStr'])

    cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim1
