import pandas as pd

def validate_sms_data(df: pd.DataFrame) -> bool:
    """
    Validates cleaned dataframe meets schema and integrity constraints:
    - Must contain 'text' and 'label' columns
    - 'label' must only be 0 or 1
    - 'text' must be non-empty strings
    - Dataset must contain both positive and negative examples
    """
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("DataFrame must contain 'label' and 'text' columns.")
        
    if df.empty:
        raise ValueError("DataFrame is empty.")
        
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1. Found: {unique_labels}")
        
    if len(unique_labels) < 2:
        raise ValueError("Dataset must contain both positive (spam) and negative (ham) samples.")
        
    if df["text"].isna().any():
        raise ValueError("Found NaN in text column.")
        
    empty_texts = (df["text"].astype(str).str.strip() == "").sum()
    if empty_texts > 0:
        raise ValueError(f"Found {empty_texts} empty string texts in dataset.")
        
    return True
