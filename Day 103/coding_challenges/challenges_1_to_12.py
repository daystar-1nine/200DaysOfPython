"""Implementation of all 12 Coding Challenges for Day 103."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from sklearn.decomposition import PCA

# C1: Cosine similarity
def c1_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

# C2: Build vocabulary
def c2_build_vocab(tokens_list):
    vocab = sorted(list({t for doc in tokens_list for t in doc}))
    return {w: i for i, w in enumerate(vocab)}

# C3: Words to integer IDs
def c3_words_to_ids(tokens, vocab):
    return [vocab[t] for t in tokens if t in vocab]

# C4: Skip-gram context pairs
def c4_skipgram_pairs(ids, window=1):
    pairs = []
    for i, t in enumerate(ids):
        for j in range(max(0, i - window), min(len(ids), i + window + 1)):
            if i != j:
                pairs.append((t, ids[j]))
    return pairs

# C5: Stable sigmoid
def c5_sigmoid(z):
    z = np.clip(z, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-z))

# C6: Negative sampling
def c6_sample_negatives(vocab_size, num_samples, target, context, seed=42):
    rng = np.random.default_rng(seed)
    negs = []
    while len(negs) < num_samples:
        s = int(rng.integers(0, vocab_size))
        if s != target and s != context:
            negs.append(s)
    return negs

# C7: Forward pass
def c7_forward(target_vec, context_vec, neg_vecs):
    pos_dot = np.dot(target_vec, context_vec)
    neg_dots = np.dot(neg_vecs, target_vec)
    return c5_sigmoid(pos_dot), c5_sigmoid(neg_dots)

# C8: Loss
def c8_loss(pos_prob, neg_probs, eps=1e-12):
    return float(-np.log(np.clip(pos_prob, eps, 1.0)) - np.sum(np.log(np.clip(1.0 - neg_probs, eps, 1.0))))

# C9: One gradient update
def c9_gradient_update(target_vec, context_vec, neg_vecs, pos_prob, neg_probs, lr=0.01):
    pos_err = pos_prob - 1.0
    neg_errs = neg_probs
    grad_context = pos_err * target_vec
    grad_neg = neg_errs[:, None] * target_vec[None, :]
    grad_target = pos_err * context_vec + np.sum(neg_errs[:, None] * neg_vecs, axis=0)
    
    new_target = target_vec - lr * grad_target
    new_context = context_vec - lr * grad_context
    new_negs = neg_vecs - lr * grad_neg
    return new_target, new_context, new_negs

# C10: Nearest neighbor search
def c10_nearest_neighbors(W, target_id, top_k=3):
    vec = W[target_id]
    scores = [(i, c1_cosine_sim(vec, W[i])) for i in range(len(W)) if i != target_id]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# C11: Train a tiny embedding model
def c11_train_tiny_model():
    corpus = [["cat", "drinks", "milk"], ["dog", "drinks", "milk"]]
    vocab = c2_build_vocab(corpus)
    pairs = []
    for doc in corpus:
        ids = c3_words_to_ids(doc, vocab)
        pairs.extend(c4_skipgram_pairs(ids, window=1))
        
    dim = 4
    V = len(vocab)
    rng = np.random.default_rng(42)
    W_in = rng.uniform(-0.5, 0.5, (V, dim))
    W_out = rng.uniform(-0.5, 0.5, (V, dim))
    
    initial_loss = None
    final_loss = None
    
    for epoch in range(20):
        epoch_loss = 0.0
        for t, c in pairs:
            negs = c6_sample_negatives(V, 2, t, c)
            pos_p, neg_p = c7_forward(W_in[t], W_out[c], W_out[negs])
            loss = c8_loss(pos_p, neg_p)
            epoch_loss += loss
            
            # Update
            new_t, new_c, new_n = c9_gradient_update(W_in[t], W_out[c], W_out[negs], pos_p, neg_p, lr=0.05)
            W_in[t] = new_t
            W_out[c] = new_c
            W_out[negs] = new_n
            
        if initial_loss is None:
            initial_loss = epoch_loss / len(pairs)
        final_loss = epoch_loss / len(pairs)
        
    return W_in, vocab, initial_loss, final_loss

# C12: PCA Visualization
def c12_pca_coords(W):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(W)

def run_all():
    print("=== EXECUTING ALL 12 CODING CHALLENGES ===")
    
    # C1
    assert np.isclose(c1_cosine_sim(np.array([1, 0]), np.array([1, 0])), 1.0)
    # C2 & C3
    vocab = c2_build_vocab([["cat", "dog"]])
    assert c3_words_to_ids(["cat", "dog"], vocab) == [0, 1]
    # C4
    assert len(c4_skipgram_pairs([0, 1, 2], 1)) == 4
    # C5
    assert np.isclose(c5_sigmoid(0.0), 0.5)
    # C6
    negs = c6_sample_negatives(5, 2, 0, 1)
    assert len(negs) == 2 and 0 not in negs and 1 not in negs
    # C7 & C8
    t_v = np.array([0.5, -0.5])
    c_v = np.array([0.5, -0.5])
    n_v = np.array([[-0.5, 0.5]])
    pos_p, neg_p = c7_forward(t_v, c_v, n_v)
    loss = c8_loss(pos_p, neg_p)
    assert loss > 0.0
    # C9
    nt, nc, nn = c9_gradient_update(t_v, c_v, n_v, pos_p, neg_p)
    assert nt.shape == (2,)
    # C10 & C11
    W, vocab, init_loss, fin_loss = c11_train_tiny_model()
    assert fin_loss < init_loss, f"Loss did not decrease: {init_loss} -> {fin_loss}"
    nn_cat = c10_nearest_neighbors(W, vocab["cat"])
    assert len(nn_cat) > 0
    # C12
    coords = c12_pca_coords(W)
    assert coords.shape == (len(vocab), 2)
    
    print(f"Tiny Model Trained Successfully: Initial Loss={init_loss:.4f} -> Final Loss={fin_loss:.4f}")
    print("[SUCCESS] All 12 Coding Challenges verified!")

if __name__ == "__main__":
    run_all()
