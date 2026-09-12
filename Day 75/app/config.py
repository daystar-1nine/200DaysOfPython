import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH_RAW: str = os.path.join(BASE_DIR, "data", "raw", "advertising_sales.csv")
    DATA_PATH_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed", "advertising_sales_processed.csv")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    CHARTS_DIR: str = os.path.join(OUTPUT_DIR, "charts")
    MODEL_RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "model_results.csv")
    CV_RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "cv_results.csv")
    HYPERPARAM_RESULTS_CSV: str = os.path.join(OUTPUT_DIR, "hyperparam_results.csv")
    COEFFICIENTS_CSV: str = os.path.join(OUTPUT_DIR, "coefficients.csv")
    PREDICTIONS_CSV: str = os.path.join(OUTPUT_DIR, "predictions.csv")
    REPORT_TXT: str = os.path.join(OUTPUT_DIR, "regression_report.txt")

    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: ["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount", "Quantity"])
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: ["Region", "Category"])
    TARGET_COL: str = "Sales"
    
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    ALPHAS: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    MAX_POLY_DEGREE: int = 5
    
    VALID_REGIONS: List[str] = field(default_factory=lambda: ["North", "South", "East", "West"])
    VALID_CATEGORIES: List[str] = field(default_factory=lambda: ["Electronics", "Clothing", "Food", "Home", "Sports"])
