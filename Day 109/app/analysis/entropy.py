import numpy as np

def compute_attention_entropy(weights: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Computes Shannon attention entropy for each sequence in the batch:
      H(A) = - sum_i A_i * log(A_i + eps)
    Higher entropy indicates diffuse/spread attention; lower entropy indicates concentrated attention.
    
    weights: (batch_size, seq_len) or (seq_len,)
    returns: (batch_size,) or float
    """
    weights = np.asarray(weights)
    is_1d = (weights.ndim == 1)
    if is_1d:
        weights = np.expand_dims(weights, 0)
        
    entropy = -np.sum(weights * np.log(weights + eps), axis=-1)
    
    if is_1d:
        return float(entropy[0])
    return entropy

def summarize_attention_distribution(weights: np.ndarray) -> dict:
    """
    Calculates summary statistics across attention weights:
      min, max, mean, standard deviation, and entropy.
    """
    weights = np.asarray(weights)
    ent = compute_attention_entropy(weights)
    return {
        "max_attention": float(np.max(weights)),
        "min_attention": float(np.min(weights)),
        "mean_attention": float(np.mean(weights)),
        "std_attention": float(np.std(weights)),
        "entropy": float(np.mean(ent)) if isinstance(ent, np.ndarray) else float(ent)
    }
