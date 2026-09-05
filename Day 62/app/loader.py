"""
Data Ingestion and Schema Validation Module for Day 62.
Loads e-commerce transactional data, parses dates, and verifies required fields.
"""

from pathlib import Path
import pandas as pd
from app.config import REQUIRED_COLUMNS


def load_ecommerce_data(filepath: Path) -> pd.DataFrame:
    """
    Ingests e-commerce CSV, validates schema integrity, and parses datetime timestamps.

    # What is used: pd.read_csv with schema validation and datetime parsing
    # Why it is used: Ensures production reliability before entering the visualization pipeline
    # How it works: Confirms file existence, verifies presence of all required columns, and derives Year_Month
    """
    if not filepath.exists():
        raise FileNotFoundError(f"E-commerce dataset not found at expected path: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("E-commerce dataset is empty. Cannot generate visual dashboard.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Dataset missing required schema columns: {missing_cols}")

    # Parse Order_Date
    if not pd.api.types.is_datetime64_any_dtype(df["Order_Date"]):
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

    df = df.dropna(subset=["Order_Date", "Revenue"]).copy()

    # Chronological grouping column
    if "Year_Month" not in df.columns:
        df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)

    return df