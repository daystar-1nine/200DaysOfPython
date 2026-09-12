import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH_RAW: str = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")
    DATA_PATH_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed", "customer_churn_cleaned.csv")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
    CHARTS_DIR: str = os.path.join(OUTPUT_DIR, "charts")
    
    PREDICTIONS_CSV: str = os.path.join(OUTPUT_DIR, "predictions.csv")
    METRICS_CSV: str = os.path.join(OUTPUT_DIR, "metrics.csv")
    CONFUSION_MATRIX_CSV: str = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
    THRESHOLD_ANALYSIS_CSV: str = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")
    COEFFICIENTS_CSV: str = os.path.join(OUTPUT_DIR, "coefficients.csv")
    REPORT_TXT: str = os.path.join(OUTPUT_DIR, "churn_report.txt")
    
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: ["Age", "Tenure_Months", "Monthly_Charges", "Total_Charges", "Support_Calls", "Late_Payments", "Usage_Hours", "Discount"])
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: ["Gender", "Contract_Type", "Internet_Service", "Payment_Method"])
    TARGET_COL: str = "Churn"
    
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    DEFAULT_THRESHOLD: float = 0.50
    
    FN_COST: float = 5000.0
    FP_COST: float = 500.0
    
    VALID_GENDERS: List[str] = field(default_factory=lambda: ["Male", "Female"])
    VALID_CONTRACTS: List[str] = field(default_factory=lambda: ["Month-to-month", "One year", "Two year"])
    VALID_INTERNET: List[str] = field(default_factory=lambda: ["DSL", "Fiber optic", "No"])
    
    def __post_init__(self):
        os.makedirs(os.path.dirname(self.DATA_PATH_PROCESSED), exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHARTS_DIR, exist_ok=True)
