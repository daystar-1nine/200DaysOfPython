from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path: str = os.path.join(BASE_DIR, 'data', 'raw', 'customer_churn.csv')
    processed_train_path: str = os.path.join(BASE_DIR, 'data', 'processed', 'train.csv')
    processed_test_path: str = os.path.join(BASE_DIR, 'data', 'processed', 'test.csv')
    output_dir: str = os.path.join(BASE_DIR, 'output')
    charts_dir: str = os.path.join(BASE_DIR, 'output', 'charts')
    metrics_dir: str = os.path.join(BASE_DIR, 'output', 'metrics')
    test_size: float = 0.20
    random_state: int = 42
    cost_fp: float = 300.0   # Retention incentive / marketing outreach cost
    cost_fn: float = 2000.0  # Revenue loss from unintercepted churner
    target_column: str = 'Churn'
    id_column: str = 'Customer_ID'
    
    numeric_features: List[str] = field(default_factory=lambda: [
        'Age', 'Tenure_Months', 'Monthly_Charges', 'Total_Charges',
        'Support_Calls', 'Late_Payments', 'Usage_Hours', 'Discount',
        'Complaints', 'Tenure_to_Age_Ratio', 'Charges_Deviation', 'Support_Per_Month'
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        'Gender', 'Contract_Type', 'Internet_Service', 'Payment_Method'
    ])
    
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    rf_max_features: str = 'sqrt'
    rf_criterion: str = 'gini'
    rf_bootstrap: bool = True
    rf_oob_score: bool = True
