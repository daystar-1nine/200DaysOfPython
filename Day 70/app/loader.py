"""
CSV Data loader and parser for hypothesis testing scenarios.
"""

import os
import pandas as pd

try:
    from app.config import config
except (ImportError, ModuleNotFoundError):
    from config import config

class DataLoader:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or config.DATA_DIR

    def load_delivery_times(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "delivery_times.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset at {path}")
        return pd.read_csv(path)

    def load_conversion_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "conversion_data.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset at {path}")
        return pd.read_csv(path)

    def load_manufacturing(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "manufacturing.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset at {path}")
        return pd.read_csv(path)

    def load_customer_orders(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "customer_orders.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset at {path}")
        return pd.read_csv(path)
