import numpy as np
import pandas as pd

import os
import pickle

from src.models.recommender import cosine_sim1, cosine_sim2, cosine_sim3

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="recommend.log")


if __name__ == "__main__":
    try:
        sim1 = cosine_sim1.get_cosine_sim1(logger)
        sim2 = cosine_sim2.get_cosine_sim2(logger)
        location_df, sim3 = cosine_sim3.get_cosine_sim3(logger)
        logger.info("Loaded all the similarity matrices")

        # Save
        logger.info("Saving similarities")
        path = os.path.join("models", "recommend")
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "cosine_sim1.pkl"), "wb") as f:
            pickle.dump(sim1, f)

        with open(os.path.join(path, "cosine_sim2.pkl"), "wb") as f:
            pickle.dump(sim2, f)

        with open(os.path.join(path, "cosine_sim3.pkl"), "wb") as f:
            pickle.dump(sim3, f)

        with open(os.path.join(path, "location_df.pkl"), "wb") as f:
            pickle.dump(location_df, f)



    except Exception as e:
        logger.error("Error loading data: %s", e)
