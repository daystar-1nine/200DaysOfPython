"""
Central Configuration for Day 62 E-Commerce Dashboard Engine.
Defines filepaths, dimensions, DPI, corporate palettes, and schema rules.
"""

from pathlib import Path

# Filepaths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
ECOMMERCE_CSV_PATH = DATA_DIR / "ecommerce_sales.csv"
DASHBOARD_PATH = OUTPUT_DIR / "ecommerce_dashboard.png"

# Visualization Parameters
DPI = 300
FIGSIZE_DASHBOARD = (16, 12)
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_HORIZONTAL = (10, 6.5)

# Corporate Styling Palettes
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_SUCCESS = "#2ca02c"
COLOR_DANGER = "#d62728"
COLOR_PURPLE = "#9467bd"

PALETTE_REGIONAL = ["#1b4f72", "#2e86c1", "#5dade2", "#aed6f1"]
PALETTE_CATEGORICAL = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

# Required Schema
REQUIRED_COLUMNS = [
    "Order_ID",
    "Order_Date",
    "Customer_ID",
    "Customer_Name",
    "Region",
    "Category",
    "Product",
    "Quantity",
    "Unit_Price",
    "Cost_Price",
    "Discount",
    "Revenue",
    "Cost",
    "Profit"
]