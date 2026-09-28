import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_sms_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    clean_df = df.copy()
    if "v1" in clean_df.columns and "v2" in clean_df.columns:
        clean_df = clean_df.rename(columns={"v1": "label", "v2": "text"})
    clean_df = clean_df.dropna(subset=["label", "text"])
    
    label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    clean_df["label"] = clean_df["label"].astype(str).str.lower().str.strip().map(label_map)
    clean_df = clean_df.dropna(subset=["label"])
    clean_df["label"] = clean_df["label"].astype(int)
    
    clean_df["text"] = clean_df["text"].astype(str).str.strip()
    clean_df = clean_df[clean_df["text"].str.len() > 0]
    
    if drop_duplicates:
        initial_len = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["text"])
        logger.info(f"Dropped {initial_len - len(clean_df)} duplicate messages.")
        
    return clean_df.reset_index(drop=True)
