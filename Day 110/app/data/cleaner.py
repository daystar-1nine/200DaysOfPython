"""
Data cleaning and preprocessing routines for SMS Spam dataset.
"""
import pandas as pd


def clean_sms_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names, maps labels, removes duplicates and missing values.
    """
    df = df.copy()

    # Handle standard Kaggle / UCI SMS Spam collection columns
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    elif "Category" in df.columns and "Message" in df.columns:
        df = df[["Category", "Message"]].rename(columns={"Category": "label", "Message": "text"})
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]]
    else:
        # Fall back to first two columns
        cols = list(df.columns[:2])
        df = df[cols].rename(columns={cols[0]: "label", cols[1]: "text"})

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()

    # Map labels to binary integer (0 = ham, 1 = spam)
    label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    df["label"] = df["label"].astype(str).str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Filter out empty text and drop duplicates
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df
