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
    MODEL_METRICS_CSV: str = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    CV_RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "cross_validation_results.csv")
    HYPERPARAM_RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "hyperparameter_results.csv")
    FEATURE_IMPORTANCE_CSV: str = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    PERMUTATION_IMPORTANCE_CSV: str = os.path.join(OUTPUT_DIR, "permutation_importance.csv")
    DECISION_RULES_TXT: str = os.path.join(OUTPUT_DIR, "decision_rules.txt")
    THRESHOLD_ANALYSIS_CSV: str = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")
    BUSINESS_COST_CSV: str = os.path.join(OUTPUT_DIR, "business_cost_analysis.csv")
    
    # Feature configurations
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        "Age", "Tenure_Months", "Monthly_Charges", "Total_Charges",
        "Support_Calls", "Late_Payments", "Usage_Hours", "Discount", "Complaints",
        "Charges_Per_Month_Ratio", "Support_Per_Tenure", "Complaint_Risk_Index",
        "Net_Monthly_Charges", "Tenure_Age_Ratio"
    ])
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        "Gender", "Contract_Type", "Internet_Service", "Payment_Method"
    ])
    TARGET_COL: str = "Churn"
    
    # Splitting & Hyperparameters
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    N_SPLITS_CV: int = 5
    DEFAULT_THRESHOLD: float = 0.50
    
    # Cost values
    FP_COST: float = 300.0   # Unnecessary retention discount cost (INR)
    FN_COST: float = 5000.0  # Lost customer lifetime value (INR)
    
    # Validation constraints
    VALID_GENDERS: List[str] = field(default_factory=lambda: ["Male", "Female"])
    VALID_CONTRACTS: List[str] = field(default_factory=lambda: ["Month-to-month", "One year", "Two year"])
    VALID_INTERNET: List[str] = field(default_factory=lambda: ["Fiber optic", "DSL", "No"])
    VALID_PAYMENTS: List[str] = field(default_factory=lambda: [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    def __post_init__(self):
        os.makedirs(os.path.dirname(self.DATA_PATH_PROCESSED), exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHARTS_DIR, exist_ok=True)
