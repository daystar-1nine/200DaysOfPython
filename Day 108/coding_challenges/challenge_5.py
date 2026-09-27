"""
Challenge 5: compare_models(results) producing model, parameters, training_time, F1, ROC-AUC.
"""
import pandas as pd

def compare_models(results: list) -> pd.DataFrame:
    """
    Produces formatted comparison summary.
    """
    df = pd.DataFrame(results)
    cols = ["model", "parameters", "training_time", "F1", "ROC-AUC"]
    return df[cols]

if __name__ == "__main__":
    mock_results = [
        {"model": "SimpleRNN", "parameters": 37377, "training_time": 0.32, "F1": 0.78, "ROC-AUC": 0.65},
        {"model": "LSTM", "parameters": 62337, "training_time": 0.53, "F1": 0.78, "ROC-AUC": 1.00},
        {"model": "GRU", "parameters": 54017, "training_time": 0.92, "F1": 1.00, "ROC-AUC": 1.00}
    ]
    df_comp = compare_models(mock_results)
    print(df_comp.to_string(index=False))
