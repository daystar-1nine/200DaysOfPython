import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_sms_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Cleans raw SMS data:
    - Strips whitespace
    - Maps labels: 'ham' -> 0, 'spam' -> 1
    - Drops missing values
    - Optionally drops duplicates and resets index
    """
    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=["label", "text"])
    
    # Map label to binary integer
    label_map = {"ham": 0, "spam": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    clean_df["label"] = clean_df["label"].astype(str).str.lower().str.strip().map(label_map)
    clean_df = clean_df.dropna(subset=["label"])
    clean_df["label"] = clean_df["label"].astype(int)
    
    # Clean text column
    clean_df["text"] = clean_df["text"].astype(str).str.strip()
    clean_df = clean_df[clean_df["text"].str.len() > 0]
    
    if drop_duplicates:
        initial_len = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["text"])
        logger.info(f"Removed {initial_len - len(clean_df)} duplicate messages.")
        
    clean_df = clean_df.reset_index(drop=True)
    return clean_df
