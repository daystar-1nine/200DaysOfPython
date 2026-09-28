import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from pathlib import Path

def extract_top_attention_tokens(tokens: List[str], weights: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    """
    Returns top k tokens with the highest attention weights for a single message.
    """
    valid_len = min(len(tokens), len(weights))
    paired = [(tokens[i], float(weights[i])) for i in range(valid_len)]
    # Sort descending by weight
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired[:k]

def aggregate_token_attention(
    tokenized_texts: List[List[str]],
    attention_weights: np.ndarray
) -> pd.DataFrame:
    """
    Aggregates average attention weight per unique token across all evaluated messages.
    """
    token_sum = defaultdict(float)
    token_count = defaultdict(int)
    
    for i, tokens in enumerate(tokenized_texts):
        w = attention_weights[i]
        for t_idx, tok in enumerate(tokens):
            if t_idx < len(w):
                token_sum[tok] += float(w[t_idx])
                token_count[tok] += 1
                
    records = []
    for tok, total in token_sum.items():
        count = token_count[tok]
        records.append({
            "token": tok,
            "average_attention": total / count,
            "occurrence_count": count
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="average_attention", ascending=False).reset_index(drop=True)
    return df
