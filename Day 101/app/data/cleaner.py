"""Dataset cleaning and validation module."""
import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans dataframe by removing nulls, empty strings, and duplicates."""
    initial_len = len(df)
    cleaned = df.dropna(subset=["text", "label"]).copy()
    
    # Ensure string types
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    
    # Filter empty texts
    cleaned = cleaned[cleaned["text"].str.len() > 0]
    
    # Standardize labels
    valid_labels = {"ham", "spam"}
    cleaned = cleaned[cleaned["label"].isin(valid_labels)]
    
    # Calculate text length and word count metadata
    cleaned["char_length"] = cleaned["text"].apply(len)
    cleaned["word_count"] = cleaned["text"].apply(lambda t: len(t.split()))
    
    return cleaned.reset_index(drop=True)
