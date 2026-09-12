import pandas as pd

try:
    from app.config import AppConfig
except ImportError:
    from config import AppConfig

from typing import Optional

def validate_data(df: pd.DataFrame, config: Optional[AppConfig] = None, min_rows: Optional[int] = None) -> bool:
    """
    Validates cleaned DataFrame. Raises ValueError on failure, returns True on success.
    """
    if config is None:
        config = AppConfig()

    issues = []

    target_col = config.TARGET_COL if config.TARGET_COL in df.columns else "Sales"
    if target_col in df.columns and (df[target_col] < 0).any():
        issues.append(f"negative {target_col} values found.")

    if "Discount" in df.columns:
        if (df["Discount"] < 0).any() or (df["Discount"] > 100).any():
            issues.append("invalid discount values (must be 0-100).")

    if min_rows is not None and len(df) < min_rows:
        issues.append(f"too few rows found ({len(df)} < {min_rows}).")
    elif len(df) == 0:
        issues.append("dataset contains zero rows.")

    if issues:
        raise ValueError(f"Validation failed with issues: {'; '.join(issues)}")

    return True
