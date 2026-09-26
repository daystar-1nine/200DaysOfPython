import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the SMS Spam dataset, maps labels to binary integers,
    removes duplicates, and drops missing values.
    """
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath, encoding='latin-1', usecols=[0, 1])
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise
    
    # Rename columns based on standard structure (v1: label, v2: text)
    df.columns = ['label', 'text']
    
    # Map labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Drop NAs
    df.dropna(subset=['label', 'text'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    # Remove duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {initial_len - len(df)} duplicates. Remaining rows: {len(df)}")
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    return df
