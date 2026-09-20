"""Complete Word2Vec model implementation using NumPy."""
import numpy as np
from typing import List, Dict, Optional
from app.data.vocabulary import Vocabulary
from app.preprocessing.context_pairs import generate_all_pairs
from app.embeddings.initialization import initialize_embeddings
from app.embeddings.negative_sampling import NegativeSampler
from app.embeddings.skipgram import train_pair_step
from app.similarity.cosine import cosine_similarity

class Word2Vec:
    """
    Educational Word2Vec (Skip-Gram with Negative Sampling) from scratch using NumPy.
    """
    def __init__(
        self,
        embedding_dim: int = 16,
        window_size: int = 2,
        negative_samples: int = 4,
        learning_rate: float = 0.025,
        epochs: int = 50,
        seed: int = 42
    ):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        
        self.vocab: Optional[Vocabulary] = None
        self.W_in: Optional[np.ndarray] = None
        self.W_out: Optional[np.ndarray] = None
        self.epoch_losses: List[float] = []

    def fit(self, tokenized_corpus: List[List[str]]) -> "Word2Vec":
        """Trains word embeddings on tokenized corpus."""
        # 1. Build Vocabulary
        self.vocab = Vocabulary().build_vocab(tokenized_corpus)
        vocab_size = len(self.vocab)
        if vocab_size < 2:
            raise ValueError("Corpus must contain at least 2 unique words.")
            
        # 2. Initialize Embeddings
        self.W_in, self.W_out = initialize_embeddings(
            vocab_size, self.embedding_dim, seed=self.seed
        )
        
        # 3. Generate Training Pairs
        pairs = generate_all_pairs(tokenized_corpus, self.vocab, window_size=self.window_size)
        if not pairs:
            raise ValueError("No context pairs generated from corpus.")
            
        # 4. Negative Sampler
        sampler = NegativeSampler(self.vocab, seed=self.seed)
        
        # 5. Training Loop
        rng = np.random.default_rng(self.seed)
        total_steps = self.epochs * len(pairs)
        current_step = 0
        self.epoch_losses = []
        
        for epoch in range(self.epochs):
            rng.shuffle(pairs)
            epoch_loss = 0.0
            
            for target_id, context_id in pairs:
                current_step += 1
                # Decay learning rate
                lr = max(0.0001, self.learning_rate * (1.0 - current_step / total_steps))
                
                # Sample negatives
                neg_ids = sampler.sample(self.negative_samples, target_id, context_id)
                
                # Train step
                step_loss = train_pair_step(
                    target_id, context_id, neg_ids, self.W_in, self.W_out, lr
                )
                epoch_loss += step_loss
                
            avg_loss = epoch_loss / len(pairs)
            self.epoch_losses.append(avg_loss)
            
        return self

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Returns the embedding vector for a given word."""
        if self.vocab is None or self.W_in is None:
            raise ValueError("Model is not fitted yet.")
        idx = self.vocab.get_id(word)
        return self.W_in[idx].copy() if idx is not None else None

    def most_similar(self, word: str, top_k: int = 5) -> List[tuple]:
        """Finds top-K nearest neighbors using cosine similarity."""
        target_vec = self.get_vector(word)
        if target_vec is None:
            return []
            
        target_id = self.vocab.get_id(word)
        scores = []
        for other_word, other_id in self.vocab.word2idx.items():
            if other_id != target_id:
                sim = cosine_similarity(target_vec, self.W_in[other_id])
                scores.append((other_word, float(sim)))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
