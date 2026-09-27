import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from app.analysis.error_analysis import perform_error_analysis
from app.analysis.model_comparison import create_model_comparison_table
from app.analysis.efficiency import compute_model_efficiency

def test_create_model_comparison_table():
    records = [
        {"model": "RNN", "parameters": 1000, "training_time_sec": 1.2, "f1": 0.85},
        {"model": "GRU", "parameters": 2500, "training_time_sec": 2.1, "f1": 0.92}
    ]
    df = create_model_comparison_table(records)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "model" in df.columns
    assert "f1" in df.columns

def test_compute_model_efficiency():
    df_base = pd.DataFrame({
        "model": ["GRU", "LSTM"],
        "parameters": [20000, 26000],
        "training_time_sec": [2.0, 3.0],
        "f1": [0.90, 0.91]
    })
    eff_df = compute_model_efficiency(df_base)
    assert "f1_per_100k_params" in eff_df.columns
    assert "f1_per_second" in eff_df.columns
    assert eff_df["f1_per_100k_params"].iloc[0] > eff_df["f1_per_100k_params"].iloc[1]

def test_perform_error_analysis(tmp_path):
    texts = ["msg 0", "msg 1", "msg 2", "msg 3"]
    y_true = np.array([0, 0, 1, 1])
    # Model probabilities:
    # RNN gets index 1 wrong (FP) and index 2 wrong (FN)
    rnn_probs = np.array([0.1, 0.8, 0.2, 0.9])
    # GRU gets only index 1 wrong (FP), gets index 2 right
    gru_probs = np.array([0.1, 0.8, 0.8, 0.9])
    
    res = perform_error_analysis(
        test_texts=texts,
        y_true=y_true,
        model_predictions={"RNN": (rnn_probs >= 0.5).astype(int), "GRU": (gru_probs >= 0.5).astype(int)},
        model_probabilities={"RNN": rnn_probs, "GRU": gru_probs},
        output_dir=tmp_path
    )
    
    assert "error_dfs" in res
    assert "comparative_summary" in res
    assert (tmp_path / "rnn_errors.csv").exists()
    assert (tmp_path / "gru_errors.csv").exists()
    
    # Verify index 2 is in RNN errors, but not GRU errors
    rnn_err = res["error_dfs"]["RNN"]
    gru_err = res["error_dfs"]["GRU"]
    assert "msg 2" in rnn_err["message"].values
    assert "msg 2" not in gru_err["message"].values
