
import os
class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHURN_DATA = os.path.join(BASE_DIR, 'data', 'raw', 'customer_churn.csv')
    MESSAGE_DATA = os.path.join(BASE_DIR, 'data', 'raw', 'support_messages.csv')
    CHARTS_DIR = os.path.join(BASE_DIR, 'output', 'charts')
