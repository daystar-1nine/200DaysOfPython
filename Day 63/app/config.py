"""
Application Configuration & Visual Styling Parameters
====================================================
Centralizes typography, color palettes, DPI, and filesystem paths.
"""

import os

# Base paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ecommerce_sales.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
REPORT_PATH = os.path.join(OUTPUT_DIR, "eda_visualization_report.txt")

# Rendering parameters
DEFAULT_DPI = 300
FIGURE_FORMAT = "png"
SNS_THEME = "whitegrid"
PRIMARY_PALETTE = "deep"
CATEGORICAL_PALETTE = "Set2"
DIVERGING_PALETTE = "coolwarm"

# Categorical hierarchy
CATEGORIES = ["Electronics", "Furniture", "Fitness", "Kitchenware", "Apparel"]
REGIONS = ["North", "South", "East", "West"]
CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Home Office"]

# Numerical feature columns for correlation and pairwise analysis
NUMERIC_FEATURES = [
    "Quantity",
    "Unit_Price",
    "Cost_Price",
    "Discount",
    "Revenue",
    "Cost",
    "Profit",
    "Profit_Margin"
]
