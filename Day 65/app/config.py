"""
Central configuration for Day 65 Statistical Analysis Engine.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ecommerce_sales.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

NUMERICAL_COLS = [
    "Quantity",
    "Unit_Price",
    "Cost_Price",
    "Discount",
    "Revenue",
    "Cost",
    "Profit"
]

PERCENTILES_LIST = [10, 25, 50, 75, 90, 95, 99]
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
FIGURE_DPI = 300
