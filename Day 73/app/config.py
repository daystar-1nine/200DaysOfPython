"""
Day 73 - Application Configuration
Defines file paths, column constants, and model hyperparameters.
"""
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    # Base Directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH_RAW: str = os.path.join(BASE_DIR, "data", "raw", "advertising_sales.csv")
    DATA_PATH_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed", "cleaned_sales.csv")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    CHARTS_DIR: str = os.path.join(OUTPUT_DIR, "charts")
    
    # Export File Paths
    REGRESSION_CSV: str = os.path.join(OUTPUT_DIR, "regression_results.csv")
    PREDICTIONS_CSV: str = os.path.join(OUTPUT_DIR, "predictions.csv")
    REPORT_TXT: str = os.path.join(OUTPUT_DIR, "regression_report.txt")
    
    # Feature & Target Columns
    FEATURE_COL: str = "Advertising_Spend"
    TARGET_COL: str = "Sales"
    
    # Modeling Hyperparameters
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42
    
    # Scenario Planning Budgets (in Rupees)
    SCENARIO_BUDGETS: List[float] = field(default_factory=lambda: [
        15000.0, 25000.0, 35000.0, 50000.0, 75000.0, 100000.0, 120000.0
    ])
