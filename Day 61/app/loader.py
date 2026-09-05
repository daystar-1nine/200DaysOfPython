"""
Data Ingestion and Schema Validation Module.
Safely loads the enterprise sales CSV, validates schema integrity, and parses datetimes.
"""

from pathlib import Path
import pandas as pd
from app.config import REQUIRED_COLUMNS


def load_sales_data(filepath: Path) -> pd.DataFrame:
    """
    Loads sales dataset from CSV, validates required columns, and enforces date parsing.

    # What is used: pd.read_csv with parse_dates and schema validation
    # Why it is used: Guards against corrupted data files and missing fields prior to analysis
    # How it works: Checks file existence, verifies all required columns, and parses Order_Date
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Sales dataset not found at expected path: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("Sales dataset is empty. Cannot perform analytical visualization.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Dataset missing required schema columns: {missing_cols}")

    # Ensure Order_Date is parsed as datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Order_Date"]):
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

    # Drop records where Order_Date or Revenue could not be parsed
    df = df.dropna(subset=["Order_Date", "Revenue"]).copy()

    # Feature engineering: extract Year_Month for consistent chronological grouping
    if "Year_Month" not in df.columns:
        df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)

    return df
