import pandas as pd
import numpy as np

def compute_model_efficiency(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes architectural efficiency metrics:
      - F1 per parameter (scaled by 10^5)
      - F1 per training second
    """
    df = comparison_df.copy()
    if "f1" in df.columns and "parameters" in df.columns:
        df["f1_per_100k_params"] = (df["f1"] / df["parameters"]) * 100000.0
    if "f1" in df.columns and "training_time_sec" in df.columns:
        df["f1_per_second"] = df["f1"] / np.maximum(df["training_time_sec"], 0.001)
    return df
