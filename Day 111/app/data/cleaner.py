"""
Data cleaning and deduplication module for SMS spam classification.
"""
import pandas as pd


def clean_sms_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes schema to 'text' and 'label' (0=ham, 1=spam).
    Strips whitespace, drops nulls, filters empty text, and removes duplicate messages.
    """
    df = df.copy()

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    elif "Category" in df.columns and "Message" in df.columns:
        df = df[["Category", "Message"]].rename(columns={"Category": "label", "Message": "text"})
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]]
    else:
        cols = list(df.columns[:2])
        df = df[cols].rename(columns={cols[0]: "label", cols[1]: "text"})

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()

    label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    df["label"] = df["label"].astype(str).str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


clean_dataset = clean_sms_data
