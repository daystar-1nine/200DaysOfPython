"""
Consolidated Coding Challenges 1 to 10 for Day 105: Neural NLP & Text Classification.
"""

from collections import Counter
from typing import Dict, List, Tuple, Union
import numpy as np


# -------------------------------------------------------------
# Challenge 1: Build a vocabulary
# -------------------------------------------------------------
def challenge_1_build_vocabulary(
    texts: List[str],
    min_freq: int = 1,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>"
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary mapping with <PAD>=0 and <UNK>=1."""
    counts = Counter(w.lower() for t in texts for w in t.split())
    sorted_words = sorted([w for w, c in counts.items() if c >= min_freq])

    word2idx = {pad_token: 0, unk_token: 1}
    idx2word = {0: pad_token, 1: unk_token}
    next_id = 2
    for w in sorted_words:
        if w not in word2idx:
            word2idx[w] = next_id
            idx2word[next_id] = w
            next_id += 1
    return word2idx, idx2word


# -------------------------------------------------------------
# Challenge 2: Encode text into IDs
# -------------------------------------------------------------
def challenge_2_encode_text(
    text: str,
    word2idx: Dict[str, int],
    unk_id: int = 1
) -> List[int]:
    """Encode string of tokens into integer IDs."""
    tokens = text.lower().split()
    return [word2idx.get(t, unk_id) for t in tokens]


# -------------------------------------------------------------
# Challenge 3: Decode IDs back into words
# -------------------------------------------------------------
def challenge_3_decode_ids(
    ids: List[int],
    idx2word: Dict[int, str],
    skip_special: bool = False
) -> List[str]:
    """Decode integer IDs back to word tokens."""
    words = []
    for i in ids:
        w = idx2word.get(i, "<UNK>")
        if skip_special and w in ("<PAD>", "<UNK>"):
            continue
        words.append(w)
    return words


# -------------------------------------------------------------
# Challenge 4: Implement padding
# -------------------------------------------------------------
def challenge_4_pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_id: int = 0
) -> List[int]:
    """Pad integer sequence to fixed max length."""
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + [pad_id] * (max_length - len(sequence))


# -------------------------------------------------------------
# Challenge 5: Implement truncation
# -------------------------------------------------------------
def challenge_5_truncate_sequence(
    sequence: List[int],
    max_length: int,
    strategy: str = "post"
) -> List[int]:
    """Truncate sequence longer than max_length."""
    if len(sequence) <= max_length:
        return sequence
    return sequence[:max_length] if strategy == "post" else sequence[-max_length:]


# -------------------------------------------------------------
# Challenge 6: Embedding lookup using NumPy
# -------------------------------------------------------------
def challenge_6_embedding_lookup(
    embedding_matrix: np.ndarray,
    token_ids: Union[List[int], np.ndarray]
) -> np.ndarray:
    """Perform vectorized row lookup from embedding weight matrix."""
    matrix = np.asarray(embedding_matrix, dtype=np.float32)
    ids = np.asarray(token_ids, dtype=np.int64)
    return matrix[ids]


# -------------------------------------------------------------
# Challenge 7: Masked mean pooling
# -------------------------------------------------------------
def challenge_7_masked_mean_pooling(
    embeddings: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Compute masked global average pooling so padding tokens are ignored."""
    emb = np.asarray(embeddings, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    m_expanded = m[:, np.newaxis]
    return np.sum(emb * m_expanded, axis=0) / max(np.sum(m), 1e-9)


# -------------------------------------------------------------
# Challenge 8: Binary Cross-Entropy manual calculation
# -------------------------------------------------------------
def challenge_8_binary_cross_entropy(
    y_true: Union[List[int], np.ndarray],
    y_pred_probs: Union[List[float], np.ndarray],
    eps: float = 1e-12
) -> float:
    """Manually calculate Binary Cross-Entropy loss.
    
    Formula: - (1/N) * sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.clip(np.asarray(y_pred_probs, dtype=np.float64).flatten(), eps, 1.0 - eps)
    bce = -np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
    return float(bce)


# -------------------------------------------------------------
# Challenge 9: Build a Neural Text Classifier from Scratch (NumPy)
# -------------------------------------------------------------
class Challenge9NumPyNeuralClassifier:
    """Minimal forward-pass neural text classifier in pure NumPy."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.embedding = rng.standard_normal((vocab_size, embedding_dim)).astype(np.float32) * 0.1
        self.W1 = rng.standard_normal((embedding_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / embedding_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim, 1)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, input_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Forward pass returning sigmoid probabilities."""
        # 1. Embedding lookup: (B, T, D)
        emb = self.embedding[input_ids]
        # 2. Masked pooling: (B, D)
        m_3d = mask[:, :, np.newaxis]
        counts = np.sum(mask, axis=1, keepdims=True)
        safe_counts = np.where(counts == 0, 1.0, counts)
        pooled = np.sum(emb * m_3d, axis=1) / safe_counts
        # 3. Dense 1 + ReLU: (B, H)
        h = np.maximum(0, np.dot(pooled, self.W1) + self.b1)
        # 4. Dense 2 + Sigmoid: (B, 1)
        logits = np.dot(h, self.W2) + self.b2
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.flatten()


# -------------------------------------------------------------
# Challenge 10: Compare Neural Model vs TF-IDF + Logistic Regression
# -------------------------------------------------------------
def challenge_10_compare():
    """Verify performance comparison between classical and neural pipelines."""
    # Dummy data
    train_texts = [
        "win money free cash prize",
        "claim your free reward today",
        "hello how are you doing",
        "can we meet tomorrow for lunch"
    ]
    train_labels = [1, 1, 0, 0]

    # Classical
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer()
    X = vec.fit_transform(train_texts)
    lr = LogisticRegression()
    lr.fit(X, train_labels)
    lr_probs = lr.predict_proba(vec.transform(["free cash win"]))[:, 1]

    # Neural
    w2i, _ = challenge_1_build_vocabulary(train_texts)
    test_ids = challenge_2_encode_text("free cash win", w2i)
    padded = challenge_4_pad_sequence(test_ids, max_length=5)
    mask = [1 if x != 0 else 0 for x in padded]

    model = Challenge9NumPyNeuralClassifier(len(w2i), embedding_dim=8, hidden_dim=4)
    neural_prob = model.forward(np.array([padded]), np.array([mask]))[0]

    return {
        "TF-IDF LogReg Spam Probability": float(lr_probs[0]),
        "Neural Classifier Spam Probability": float(neural_prob)
    }


if __name__ == "__main__":
    print("Verifying all 10 coding challenges...")

    # 1. Vocab
    w2i, i2w = challenge_1_build_vocabulary(["python is great", "python neural network"])
    assert w2i["<PAD>"] == 0 and w2i["<UNK>"] == 1
    assert "python" in w2i

    # 2. Encode
    enc = challenge_2_encode_text("python unknown_word", w2i)
    assert enc[0] == w2i["python"]
    assert enc[1] == 1  # <UNK>

    # 3. Decode
    dec = challenge_3_decode_ids(enc, i2w)
    assert dec[0] == "python" and dec[1] == "<UNK>"

    # 4. Pad
    padded = challenge_4_pad_sequence([1, 2], max_length=4)
    assert padded == [1, 2, 0, 0]

    # 5. Truncate
    trunc = challenge_5_truncate_sequence([1, 2, 3, 4, 5], max_length=3)
    assert trunc == [1, 2, 3]

    # 6. Lookup
    W = np.ones((5, 3))
    looked = challenge_6_embedding_lookup(W, [1, 3])
    assert looked.shape == (2, 3)

    # 7. Masked pooling
    embs = np.array([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]])
    pool = challenge_7_masked_mean_pooling(embs, np.array([1, 1, 0]))
    assert np.allclose(pool, [2.0, 3.0])

    # 8. BCE
    bce = challenge_8_binary_cross_entropy([1, 0], [0.9, 0.1])
    assert bce < 0.2

    # 9. Neural forward
    model = Challenge9NumPyNeuralClassifier(10, 4, 2)
    p = model.forward(np.array([[1, 2, 0]]), np.array([[1, 1, 0]]))
    assert 0.0 <= p[0] <= 1.0

    # 10. Compare
    comp = challenge_10_compare()
    assert "TF-IDF LogReg Spam Probability" in comp
    assert "Neural Classifier Spam Probability" in comp

    print("[SUCCESS] All 10 Coding Challenges Verified Successfully!")
