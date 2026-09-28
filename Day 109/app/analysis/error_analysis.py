import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from .entropy import compute_attention_entropy
from .attention_analysis import extract_top_attention_tokens

def perform_error_analysis(
    texts: List[str],
    tokenized_texts: List[List[str]],
    y_true: np.ndarray,
    gru_probs: np.ndarray,
    attn_probs: np.ndarray,
    attn_weights: np.ndarray,
    output_dir: Path,
    threshold: float = 0.5
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true).astype(int)
    gru_preds = (np.asarray(gru_probs) >= threshold).astype(int)
    attn_preds = (np.asarray(attn_probs) >= threshold).astype(int)
    
    gru_errors = (gru_preds != y_true)
    attn_errors = (attn_preds != y_true)
    
    entropies = compute_attention_entropy(attn_weights)
    
    # 1. Detailed Attention Error CSV
    records = []
    for i in np.where(attn_errors)[0]:
        top_tokens = extract_top_attention_tokens(tokenized_texts[i], attn_weights[i], k=3)
        top_str = ", ".join([f"{tok}:{wt:.3f}" for tok, wt in top_tokens])
        records.append({
            "message": texts[i],
            "actual_label": int(y_true[i]),
            "prediction": int(attn_preds[i]),
            "probability": float(attn_probs[i]),
            "attention_entropy": float(entropies[i]),
            "top_attention_tokens": top_str,
            "error_type": "False Positive" if attn_preds[i] == 1 else "False Negative"
        })
        
    df_err = pd.DataFrame(records)
    df_err.to_csv(output_dir / "attention_errors.csv", index=False)
    
    # Comparative subsets:
    # GRU ❌, Attention ✅
    gru_wrong_attn_right = np.where(gru_errors & (~attn_errors))[0]
    # GRU ✅, Attention ❌
    gru_right_attn_wrong = np.where((~gru_errors) & attn_errors)[0]
    # Both ❌
    both_wrong = np.where(gru_errors & attn_errors)[0]
    
    summary = {
        "gru_wrong_attn_right_count": len(gru_wrong_attn_right),
        "gru_right_attn_wrong_count": len(gru_right_attn_wrong),
        "both_wrong_count": len(both_wrong),
        "gru_wrong_attn_right_indices": gru_wrong_attn_right.tolist(),
        "gru_right_attn_wrong_indices": gru_right_attn_wrong.tolist(),
        "both_wrong_indices": both_wrong.tolist()
    }
    
    return {
        "error_df": df_err,
        "comparative_summary": summary
    }
