import numpy as np
import pandas as pd

import os

import src.utils as utils

# Logging configuration
logger = utils.configure_logger(__name__, log_file="preprocessing_l2.log")


def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.debug("Loading data from %s", file_path)
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error("Error loading data: %s", e)
        return pd.DataFrame()


def process_sector(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    try:
        logger.debug("Processing sector")

        df.insert(
            loc=3,
            column="sector",
            value=df["property_name"]
            .str.split("in")
            .str.get(1)
            .str.replace("Gurgaon", "")
            .str.strip(),
        )
        df["sector"] = df["sector"].str.lower()

        df["sector"] = df["sector"].str.replace("dharam colony", "sector 12")
        df["sector"] = df["sector"].str.replace("krishna colony", "sector 7")
        df["sector"] = df["sector"].str.replace("suncity", "sector 54")
        df["sector"] = df["sector"].str.replace("prem nagar", "sector 13")
        df["sector"] = df["sector"].str.replace("mg road", "sector 28")
        df["sector"] = df["sector"].str.replace("gandhi nagar", "sector 28")
        df["sector"] = df["sector"].str.replace("laxmi garden", "sector 11")
        df["sector"] = df["sector"].str.replace("shakti nagar", "sector 11")
        df["sector"] = df["sector"].str.replace("baldev nagar", "sector 7")
        df["sector"] = df["sector"].str.replace("shivpuri", "sector 7")
        df["sector"] = df["sector"].str.replace("garhi harsaru", "sector 17")
        df["sector"] = df["sector"].str.replace("imt manesar", "manesar")
        df["sector"] = df["sector"].str.replace("adarsh nagar", "sector 12")
        df["sector"] = df["sector"].str.replace("shivaji nagar", "sector 11")
        df["sector"] = df["sector"].str.replace("bhim nagar", "sector 6")
        df["sector"] = df["sector"].str.replace("madanpuri", "sector 7")
        df["sector"] = df["sector"].str.replace("saraswati vihar", "sector 28")
        df["sector"] = df["sector"].str.replace("arjun nagar", "sector 8")
        df["sector"] = df["sector"].str.replace("ravi nagar", "sector 9")
        df["sector"] = df["sector"].str.replace("vishnu garden", "sector 105")
        df["sector"] = df["sector"].str.replace("bhondsi", "sector 11")
        df["sector"] = df["sector"].str.replace("surya vihar", "sector 21")
        df["sector"] = df["sector"].str.replace("devilal colony", "sector 9")
        df["sector"] = df["sector"].str.replace("valley view estate", "gwal pahari")
        df["sector"] = df["sector"].str.replace("mehrauli  road", "sector 14")
        df["sector"] = df["sector"].str.replace("jyoti park", "sector 7")
        df["sector"] = df["sector"].str.replace("ansal plaza", "sector 23")
        df["sector"] = df["sector"].str.replace("dayanand colony", "sector 6")
        df["sector"] = df["sector"].str.replace("sushant lok phase 2", "sector 55")
        df["sector"] = df["sector"].str.replace("chakkarpur", "sector 28")
        df["sector"] = df["sector"].str.replace("greenwood city", "sector 45")
        df["sector"] = df["sector"].str.replace("subhash nagar", "sector 12")
        df["sector"] = df["sector"].str.replace("sohna road road", "sohna road")
        df["sector"] = df["sector"].str.replace("malibu town", "sector 47")
        df["sector"] = df["sector"].str.replace("surat nagar 1", "sector 104")
        df["sector"] = df["sector"].str.replace("new colony", "sector 7")
        df["sector"] = df["sector"].str.replace("mianwali colony", "sector 12")
        df["sector"] = df["sector"].str.replace("jacobpura", "sector 12")
        df["sector"] = df["sector"].str.replace("rajiv nagar", "sector 13")
        df["sector"] = df["sector"].str.replace("ashok vihar", "sector 3")
        df["sector"] = df["sector"].str.replace("dlf phase 1", "sector 26")
        df["sector"] = df["sector"].str.replace("nirvana country", "sector 50")
        df["sector"] = df["sector"].str.replace("palam vihar", "sector 2")
        df["sector"] = df["sector"].str.replace("dlf phase 2", "sector 25")
        df["sector"] = df["sector"].str.replace("sushant lok phase 1", "sector 43")
        df["sector"] = df["sector"].str.replace("laxman vihar", "sector 4")
        df["sector"] = df["sector"].str.replace("dlf phase 4", "sector 28")
        df["sector"] = df["sector"].str.replace("dlf phase 3", "sector 24")
        df["sector"] = df["sector"].str.replace("sushant lok phase 3", "sector 57")
        df["sector"] = df["sector"].str.replace("dlf phase 5", "sector 43")
        df["sector"] = df["sector"].str.replace("rajendra park", "sector 105")
        df["sector"] = df["sector"].str.replace("uppals southend", "sector 49")
        df["sector"] = df["sector"].str.replace("sohna", "sohna road")
        df["sector"] = df["sector"].str.replace(
            "ashok vihar phase 3 extension", "sector 5"
        )
        df["sector"] = df["sector"].str.replace("south city 1", "sector 41")
        df["sector"] = df["sector"].str.replace("ashok vihar phase 2", "sector 5")

        a = df["sector"].value_counts()[df["sector"].value_counts() >= 3]
        df = df[df["sector"].isin(a.index)]

        df["sector"] = df["sector"].str.replace("sector 95a", "sector 95")
        df["sector"] = df["sector"].str.replace("sector 23a", "sector 23")
        df["sector"] = df["sector"].str.replace("sector 12a", "sector 12")
        df["sector"] = df["sector"].str.replace("sector 3a", "sector 3")
        df["sector"] = df["sector"].str.replace("sector 110 a", "sector 110")
        df["sector"] = df["sector"].str.replace("patel nagar", "sector 15")
        df["sector"] = df["sector"].str.replace("a block sector 43", "sector 43")
        df["sector"] = df["sector"].str.replace("maruti kunj", "sector 12")
        df["sector"] = df["sector"].str.replace("b block sector 43", "sector 43")
        df["sector"] = df["sector"].str.replace("sector-33 sohna road", "sector 33")
        df["sector"] = df["sector"].str.replace("sector 1 manesar", "manesar")
        df["sector"] = df["sector"].str.replace("sector 4 phase 2", "sector 4")
        df["sector"] = df["sector"].str.replace("sector 1a manesar", "manesar")
        df["sector"] = df["sector"].str.replace("c block sector 43", "sector 43")
        df["sector"] = df["sector"].str.replace("sector 89 a", "sector 89")
        df["sector"] = df["sector"].str.replace("sector 2 extension", "sector 2")
        df["sector"] = df["sector"].str.replace("sector 36 sohna road", "sector 36")

        return df

    except Exception as e:
        logger.error("Error processing sector: %s", e)
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        data_path = os.path.join("data", "interim")
        file_path = os.path.join(data_path, "gurgaon_properties.csv")
        df = load_data(file_path)
        if df.empty:
            raise ValueError("Data loading failed: Empty DataFrame")

        # Pipeline
        df = df.pipe(process_sector)

        # features to drop -> property_name, address, description, rating
        df.drop(
            columns=["property_name", "address", "description", "rating"], inplace=True
        )

        data_path = os.path.join("data", "interim")
        utils.save_data(
            df, data_path, "gurgaon_properties_cleaned_v1.csv", logger=logger
        )

    except Exception as e:
        logger.error("Error loading data: %s", e)
