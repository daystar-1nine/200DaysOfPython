
import os
import pandas as pd
import logging

class AppConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'customer_churn.csv')

def load_data(filepath=None):
    if filepath is None:
        filepath = AppConfig.DATA_PATH
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise
