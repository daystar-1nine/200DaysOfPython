import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def perform_error_analysis(
    test_texts: List[str],
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    model_probabilities: Dict[str, np.ndarray],
    output_dir: Path,
    threshold: float = 0.5
) -> Dict[str, pd.DataFrame]:
    """
    Performs message-level error analysis across recurrent models.
    Produces individual error CSVs and comparative case subsets:
      - All wrong: RNN ❌, LSTM ❌, GRU ❌
      - GRU wins:  RNN ❌, LSTM ❌, GRU ✅
      - GRU loses: RNN ✅, LSTM ✅, GRU ❌
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true).astype(int)
    error_dfs = {}
    error_indices = {}
    
    for model_name, probs in model_probabilities.items():
        preds = (probs >= threshold).astype(int)
        errors_mask = (preds != y_true)
        error_indices[model_name] = set(np.where(errors_mask)[0])
        
        df_errors = pd.DataFrame({
            "message": [test_texts[i] for i in np.where(errors_mask)[0]],
            "actual_label": y_true[errors_mask],
            "predicted_label": preds[errors_mask],
            "probability": probs[errors_mask],
            "error_type": [
                "False Positive (Ham as Spam)" if p == 1 else "False Negative (Spam as Ham)"
                for p in preds[errors_mask]
            ]
        })
        error_dfs[model_name] = df_errors
        
        # Save individual error files: e.g. output/gru_errors.csv
        file_name = f"{model_name.lower().replace(' ', '_')}_errors.csv"
        df_errors.to_csv(output_dir / file_name, index=False)
        
    # Comparative subsets for RNN, LSTM, GRU
    rnn_set = error_indices.get("RNN", set())
    lstm_set = error_indices.get("LSTM", set())
    gru_set = error_indices.get("GRU", set())
    
    all_wrong = rnn_set.intersection(lstm_set).intersection(gru_set)
    gru_wins = (rnn_set.intersection(lstm_set)) - gru_set
    gru_loses = gru_set - (rnn_set.union(lstm_set))
    
    comparative_summary = {
        "all_wrong_count": len(all_wrong),
        "gru_wins_count": len(gru_wins),
        "gru_loses_count": len(gru_loses),
        "all_wrong_indices": list(all_wrong),
        "gru_wins_indices": list(gru_wins),
        "gru_loses_indices": list(gru_loses)
    }
    
    return {
        "error_dfs": error_dfs,
        "comparative_summary": comparative_summary
    }
