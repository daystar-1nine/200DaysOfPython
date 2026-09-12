import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH_RAW: str = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")
    DATA_PATH_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed", "cleaned_customer_churn.csv")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    CHARTS_DIR: str = os.path.join(OUTPUT_DIR, "charts")
    
    # Output file paths
    PREDICTIONS_CSV: str = os.path.join(OUTPUT_DIR, "predictions.csv")
    BINARY_METRICS_CSV: str = os.path.join(OUTPUT_DIR, "binary_metrics.csv")
    MULTICLASS_METRICS_CSV: str = os.path.join(OUTPUT_DIR, "multiclass_metrics.csv")
    THRESHOLD_ANALYSIS_CSV: str = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")
    BUSINESS_COST_CSV: str = os.path.join(OUTPUT_DIR, "business_cost_analysis.csv")
    RISK_SCORES_CSV: str = os.path.join(OUTPUT_DIR, "risk_scores.csv")
    CLASSIFICATION_REPORT_TXT: str = os.path.join(OUTPUT_DIR, "classification_report.txt")
    
    # Feature configurations
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        "Age", "Tenure_Months", "Monthly_Charges", "Total_Charges",
        "Support_Calls", "Late_Payments", "Usage_Hours", "Discount", "Complaints",
        "Charges_Per_Month_Ratio", "Support_Per_Tenure", "Complaint_Risk_Index", "Net_Monthly_Charges"
    ])
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "Gender", "Contract_Type", "Internet_Service", "Payment_Method"
    ])
    
    TARGET_BINARY: str = "Churn"
    TARGET_MULTICLASS: str = "Risk_Level"
    
    # Model & Validation parameters
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    N_SPLITS_CV: int = 5
    DEFAULT_THRESHOLD: float = 0.50
    
    # Business costs
    FP_COST: float = 300.0   # Cost of an unnecessary retention offer (INR)
    FN_COST: float = 5000.0  # Cost of lost customer lifetime value (INR)
    
    # Domain validation values
    VALID_GENDERS: List[str] = field(default_factory=lambda: ["Male", "Female"])
    VALID_CONTRACTS: List[str] = field(default_factory=lambda: ["Month-to-month", "One year", "Two year"])
    VALID_INTERNET: List[str] = field(default_factory=lambda: ["Fiber optic", "DSL", "No"])
    VALID_PAYMENTS: List[str] = field(default_factory=lambda: [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    VALID_RISK_LEVELS: List[str] = field(default_factory=lambda: ["Low", "Medium", "High"])
    
    def __post_init__(self):
        os.makedirs(os.path.dirname(self.DATA_PATH_PROCESSED), exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHARTS_DIR, exist_ok=True)
