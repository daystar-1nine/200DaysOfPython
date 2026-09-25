"""
Coding Challenge 6: Masked Mean Pooling
Computes mean across sequence dimension while ignoring padded positions.
"""

import numpy as np


def masked_mean(sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute mean pooling over sequence ignoring zero-mask positions.
    
    Args:
        sequence: Array of shape (T, D) or (B, T, D).
        mask: Array of shape (T,) or (B, T) with 1.0 for valid, 0.0 for pad.
        
    Returns:
        Array of shape (D,) or (B, D).
    """
    if sequence.ndim == 2:
        mask_expanded = mask[:, None]
        masked_seq = sequence * mask_expanded
        valid_count = max(float(mask.sum()), 1.0)
        return masked_seq.sum(axis=0) / valid_count
    elif sequence.ndim == 3:
        mask_expanded = mask[:, :, None]
        masked_seq = sequence * mask_expanded
        valid_counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        return masked_seq.sum(axis=1) / valid_counts
    else:
        raise ValueError("Sequence must be 2D (T, D) or 3D (B, T, D).")


if __name__ == "__main__":
    seq = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [0.0, 0.0],  # padded
    ], dtype=np.float32)
    mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    pooled = masked_mean(seq, mask)
    print("Challenge 6: Masked Mean Pooling")
    print(f"  Pooled vector: {pooled}")
    # Expected: ([1, 2] + [3, 4]) / 2 = [2.0, 3.0]
    np.testing.assert_allclose(pooled, [2.0, 3.0])
    print("  [SUCCESS] Masked mean pooling verified!")
