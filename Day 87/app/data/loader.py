
import pandas as pd
from app.config import Config

def load_data():
    return pd.read_csv(Config.DATA_PATH)
