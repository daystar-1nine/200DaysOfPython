"""
Coding Challenge 105.7: Masked Mean Pooling.
Day 105: Neural NLP & Text Classification.
"""

import numpy as np


def challenge_masked_mean_pooling(
    embeddings: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Challenge 7: Implement masked mean pooling.
    
    Formula: h = sum(mask_t * e_t) / sum(mask_t)
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)

    if emb.ndim == 2:
        m_2d = m[:, np.newaxis]
        return np.sum(emb * m_2d, axis=0) / max(np.sum(m), 1e-9)
    else:
        m_3d = m[:, :, np.newaxis]
        counts = np.sum(m, axis=1, keepdims=True)
        safe_counts = np.where(counts == 0, 1.0, counts)
        return np.sum(emb * m_3d, axis=1) / safe_counts


if __name__ == "__main__":
    embeddings = np.array([
        [2.0, 4.0],
        [6.0, 8.0],
        [999.0, 999.0]  # <PAD>
    ], dtype=np.float32)
    mask = np.array([1, 1, 0], dtype=np.float32)

    pooled = challenge_masked_mean_pooling(embeddings, mask)
    print("Pooled result:", pooled)
    expected = np.array([4.0, 6.0], dtype=np.float32)
    assert np.allclose(pooled, expected), f"Expected {expected}, got {pooled}"

    print("[SUCCESS] Challenge 105.7 Passed!")
