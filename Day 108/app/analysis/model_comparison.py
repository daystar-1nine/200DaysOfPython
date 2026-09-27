import pandas as pd
from typing import List, Dict

def create_model_comparison_table(results_list: List[Dict]) -> pd.DataFrame:
    """
    Creates a unified model comparison dataframe.
    """
    df = pd.DataFrame(results_list)
    # Order columns cleanly if present
    cols = [
        "model", "parameters", "training_time_sec", "epochs_trained",
        "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"
    ]
    actual_cols = [c for c in cols if c in df.columns]
    remaining = [c for c in df.columns if c not in actual_cols]
    return df[actual_cols + remaining]
