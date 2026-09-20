"""
Document embedder module for Day 104 Semantic Search Engine.
Trains/loads word embeddings and computes document representations via pooling.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .pooling import mean_pooling, weighted_pooling


class DocumentEmbedder:
    """Manages word embeddings and converts tokenized documents into dense document vectors."""

    def __init__(
        self,
        embedding_dim: int = 64,
        unk_strategy: str = "zero",
        seed: int = 42
    ):
        self.embedding_dim = embedding_dim
        self.unk_strategy = unk_strategy
        self.seed = seed
        self.embeddings: Dict[str, np.ndarray] = {}
        self.unk_vector: Optional[np.ndarray] = None
        
        # Initialize deterministic UNK vector if needed
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        self.unk_vector = vec / (norm if norm > 0 else 1.0)

    def train_on_corpus(
        self,
        tokenized_corpus: List[List[str]],
        window_size: int = 4,
        min_count: int = 1
    ) -> "DocumentEmbedder":
        """Train dense word embeddings using PPMI co-occurrence matrix and TruncatedSVD.
        
        This implements the closed-form matrix factorization equivalent of 
        Skip-Gram word representations (Levy & Goldberg, 2014).
        
        Args:
            tokenized_corpus: List of tokenized documents (each a list of strings).
            window_size: Symmetric context window size.
            min_count: Minimum frequency for words to be included.
            
        Returns:
            self
        """
        # 1. Count word frequencies and build vocabulary
        word_counts = Counter(word for doc in tokenized_corpus for word in doc)
        vocab = [w for w, c in word_counts.items() if c >= min_count]
        vocab.sort()
        word2idx = {w: i for i, w in enumerate(vocab)}
        v_size = len(vocab)
        
        if v_size == 0:
            return self

        # 2. Build co-occurrence counts
        cooccur = defaultdict(float)
        total_tokens = 0

        for doc in tokenized_corpus:
            doc_len = len(doc)
            total_tokens += doc_len
            for i, target in enumerate(doc):
                if target not in word2idx:
                    continue
                t_idx = word2idx[target]
                start = max(0, i - window_size)
                end = min(doc_len, i + window_size + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    ctx = doc[j]
                    if ctx in word2idx:
                        c_idx = word2idx[ctx]
                        # Harmonic distance weighting
                        dist = abs(i - j)
                        cooccur[(t_idx, c_idx)] += 1.0 / dist

        if not cooccur:
            return self

        # 3. Compute PPMI matrix
        rows, cols, data = [], [], []
        row_totals = np.zeros(v_size, dtype=np.float64)
        total_cooccur = 0.0

        for (t_idx, c_idx), count in cooccur.items():
            rows.append(t_idx)
            cols.append(c_idx)
            data.append(count)
            row_totals[t_idx] += count
            total_cooccur += count

        cooccur_mat = csr_matrix((data, (rows, cols)), shape=(v_size, v_size), dtype=np.float64)

        # Compute marginals
        p_target = row_totals / total_cooccur
        p_context = np.array(cooccur_mat.sum(axis=0)).flatten() / total_cooccur
        # Shift context distribution with 0.75 exponent for negative sampling parity
        p_context = np.power(p_context, 0.75)
        p_context /= np.sum(p_context)

        # Build sparse PPMI
        ppmi_rows, ppmi_cols, ppmi_data = [], [], []
        for r, c, val in zip(rows, cols, data):
            p_tc = val / total_cooccur
            p_t = p_target[r]
            p_c = p_context[c]
            if p_t > 0 and p_c > 0 and p_tc > 0:
                pmi = np.log(p_tc / (p_t * p_c))
                if pmi > 0:
                    ppmi_rows.append(r)
                    ppmi_cols.append(c)
                    ppmi_data.append(pmi)

        ppmi_mat = csr_matrix((ppmi_data, (ppmi_rows, ppmi_cols)), shape=(v_size, v_size), dtype=np.float32)

        # 4. Truncated SVD dimensionality reduction
        n_components = min(self.embedding_dim, v_size - 1 if v_size > 1 else 1)
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        u_sigma = svd.fit_transform(ppmi_mat)

        # Pad to embedding_dim if necessary
        if u_sigma.shape[1] < self.embedding_dim:
            pad_width = self.embedding_dim - u_sigma.shape[1]
            u_sigma = np.pad(u_sigma, ((0, 0), (0, pad_width)), mode="constant")

        # 5. Populate embeddings dictionary with L2 normalized word vectors
        self.embeddings = {}
        for word, idx in word2idx.items():
            vec = u_sigma[idx].astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.embeddings[word] = vec

        return self

    def embed_document(
        self,
        tokens: List[str],
        pooling: str = "mean",
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Embed a single tokenized document.
        
        Args:
            tokens: List of word tokens.
            pooling: 'mean' or 'weighted'.
            weights: Token weight mapping (required if pooling is 'weighted').
            
        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        if pooling == "weighted" and weights is not None:
            return weighted_pooling(
                tokens=tokens,
                embeddings=self.embeddings,
                weights=weights,
                embedding_dim=self.embedding_dim,
                unk_vector=self.unk_vector,
                unk_strategy=self.unk_strategy
            )
        return mean_pooling(
            tokens=tokens,
            embeddings=self.embeddings,
            embedding_dim=self.embedding_dim,
            unk_vector=self.unk_vector,
            unk_strategy=self.unk_strategy
        )

    def embed_corpus(
        self,
        tokenized_corpus: List[List[str]],
        pooling: str = "mean",
        tfidf_model: Any = None
    ) -> np.ndarray:
        """Embed a collection of documents.
        
        Args:
            tokenized_corpus: List of token lists.
            pooling: 'mean' or 'weighted'.
            tfidf_model: Optional fitted TfidfRepresentation for weighted pooling.
            
        Returns:
            2D numpy array of shape (n_documents, embedding_dim).
        """
        doc_vectors = []
        for doc_tokens in tokenized_corpus:
            weights = None
            if pooling == "weighted" and tfidf_model is not None:
                text = " ".join(doc_tokens)
                weights = tfidf_model.get_token_weights(text)

            vec = self.embed_document(doc_tokens, pooling=pooling, weights=weights)
            doc_vectors.append(vec)

        return np.array(doc_vectors, dtype=np.float32)

    def save_embeddings(self, filepath: Union[str, Path]) -> None:
        """Save word embeddings matrix and vocabulary to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        vocab = list(self.embeddings.keys())
        matrix = np.array([self.embeddings[w] for w in vocab], dtype=np.float32)
        np.save(path, matrix)
        vocab_path = path.with_suffix(".json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f)

    def load_embeddings(self, filepath: Union[str, Path]) -> "DocumentEmbedder":
        """Load word embeddings from disk."""
        path = Path(filepath)
        vocab_path = path.with_suffix(".json")
        matrix = np.load(path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.embeddings = {w: matrix[i] for i, w in enumerate(vocab)}
        self.embedding_dim = matrix.shape[1]
        return self
