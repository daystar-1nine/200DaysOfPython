
import pandas as pd
from app.config import Config

def load_data(path=Config.DATA_PATH):
    return pd.read_csv(path)
