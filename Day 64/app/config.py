"""
Application Configuration & Visual Styling Parameters
====================================================
"""

import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ecommerce_sales.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
REPORT_PATH = os.path.join(OUTPUT_DIR, "advanced_eda_report.txt")

DEFAULT_DPI = 300
FIGURE_FORMAT = "png"
SNS_THEME = "whitegrid"
PRIMARY_PALETTE = "deep"
CATEGORICAL_PALETTE = "Set2"
DIVERGING_PALETTE = "coolwarm"

NUMERIC_FEATURES = [
    "Quantity",
    "Unit_Price",
    "Cost_Price",
    "Discount",
    "Revenue",
    "Cost",
    "Profit"
]
