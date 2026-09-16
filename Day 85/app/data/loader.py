
import pandas as pd
from app.config import Config

def load_churn_data():
    return pd.read_csv(Config.CHURN_DATA)
    
def load_message_data():
    return pd.read_csv(Config.MESSAGE_DATA)
