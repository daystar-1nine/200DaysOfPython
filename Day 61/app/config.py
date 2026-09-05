"""
Central Configuration Module for Day 61 Sales Visualization Engine.
Defines file paths, visualization palettes, figure dimensions, and validation schemas.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORT_PATH = OUTPUT_DIR / "visualization_report.txt"
SALES_CSV_PATH = DATA_DIR / "sales.csv"

# Visualization Dimensions & DPI Settings
DPI = 300
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (11, 6)
FIGSIZE_HORIZONTAL = (10, 6.5)
FIGSIZE_PIE = (8, 8)

# Corporate Visualization Color Palettes
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_SUCCESS = "#2ca02c"
COLOR_DANGER = "#d62728"
COLOR_PURPLE = "#9467bd"
COLOR_MUTED_GRAY = "#7f7f7f"

PALETTE_REGIONAL = ["#1b4f72", "#2e86c1", "#5dade2", "#aed6f1"]
PALETTE_CATEGORICAL = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

# Required Schema Columns
REQUIRED_COLUMNS = [
    "Order_Date",
    "Region",
    "Category",
    "Product",
    "Customer_Name",
    "Quantity",
    "Revenue",
    "Profit"
]
