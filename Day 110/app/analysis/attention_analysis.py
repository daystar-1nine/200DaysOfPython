"""
Attention weight extraction, head diversity, and entropy analysis for Transformers.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn


def extract_sample_attention(
    model: nn.Module,
    token_ids: np.ndarray,
    device: str = "cpu"
) -> List[np.ndarray]:
    """
    Runs a single sequence through the model in eval mode and extracts
    attention weight matrices for every layer: List of shape (num_heads, seq_len, seq_len).
    """
    model.eval()
    inp = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        probs, all_weights = model(inp)

    # all_weights is a list of tensors with shape (1, num_heads, seq_len, seq_len)
    return [w.squeeze(0).cpu().numpy() for w in all_weights]


def compute_head_entropy(attention_weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Shannon entropy for each query row across keys:
        H(p) = -sum(p * log(p + eps))
    Args:
        attention_weights: (num_heads, seq_len, seq_len)
    Returns:
        entropies: (num_heads, seq_len)
    """
    clipped = np.clip(attention_weights, eps, 1.0)
    entropy = -np.sum(attention_weights * np.log(clipped), axis=-1)
    return entropy


def compute_head_correlation(attention_weights: np.ndarray) -> np.ndarray:
    """
    Measures cosine similarity or correlation between attention matrices of different heads
    to quantify head specialization/diversity.
    Args:
        attention_weights: (num_heads, seq_len, seq_len)
    Returns:
        corr_matrix: (num_heads, num_heads)
    """
    num_heads = attention_weights.shape[0]
    flat_heads = attention_weights.reshape(num_heads, -1)
    norms = np.linalg.norm(flat_heads, axis=1, keepdims=True) + 1e-9
    normalized = flat_heads / norms
    similarity = np.dot(normalized, normalized.T)
    return similarity
