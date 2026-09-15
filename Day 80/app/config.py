import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path: str = os.path.join(BASE_DIR, 'data', 'raw', 'customer_churn.csv')
    processed_data_path: str = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_customer_churn.csv')
    output_dir: str = os.path.join(BASE_DIR, 'output')
    charts_dir: str = os.path.join(BASE_DIR, 'output', 'charts')
    
    test_size: float = 0.20
    random_state: int = 42
    cost_fp: float = 300.0   # Retention offer cost / false alarm
    cost_fn: float = 2000.0  # Churned customer lost revenue
    target_column: str = 'Churn'
    id_column: str = 'Customer_ID'
    
    numeric_features: List[str] = field(default_factory=lambda: [
        'Age', 'Tenure_Months', 'Monthly_Charges', 'Total_Charges',
        'Support_Calls', 'Late_Payments', 'Usage_Hours', 'Discount', 'Complaints'
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        'Gender', 'Contract_Type', 'Internet_Service', 'Payment_Method'
    ])
