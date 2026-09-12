import os
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH_RAW: str = os.path.join(BASE_DIR, "data", "raw", "advertising_sales.csv")
    DATA_PATH_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed", "cleaned_sales.csv")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    CHARTS_DIR: str = os.path.join(BASE_DIR, "output", "charts")
    
    # CSV exports
    PREDICTIONS_CSV: str = os.path.join(OUTPUT_DIR, "predictions.csv")
    MODEL_METRICS_CSV: str = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    COEFFICIENTS_CSV: str = os.path.join(OUTPUT_DIR, "coefficients.csv")
    VIF_REPORT_CSV: str = os.path.join(OUTPUT_DIR, "vif_report.csv")
    REGRESSION_REPORT_TXT: str = os.path.join(OUTPUT_DIR, "regression_report.txt")
    
    NUMERIC_FEATURES: list[str] = field(default_factory=lambda: ["TV_Spend", "Digital_Spend", "Radio_Spend", "Discount", "Quantity"])
    CATEGORICAL_FEATURES: list[str] = field(default_factory=lambda: ["Region", "Category"])
    TARGET_COL: str = "Sales"
    
    VALID_REGIONS: list[str] = field(default_factory=lambda: ["North", "South", "East", "West"])
    VALID_CATEGORIES: list[str] = field(default_factory=lambda: ["Electronics", "Clothing", "Food", "Home", "Sports"])
    
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    VIF_THRESHOLD: float = 10.0
