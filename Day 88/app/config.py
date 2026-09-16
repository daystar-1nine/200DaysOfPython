
import os
class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'customer_churn.csv')
    CHARTS_DIR = os.path.join(BASE_DIR, 'output', 'charts')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
