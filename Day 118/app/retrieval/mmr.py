"""
Maximal Marginal Relevance (MMR) Engine for retrieval diversification.
Balances relevance to the query with diversity among selected chunks to eliminate redundancy.
Based on Carbonell & Goldstein (SIGIR 1998).
"""
from typing import List, Optional, Sequence
import numpy as np

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk
from app.embeddings.similarity import batch_cosine_similarity, cosine_similarity
from app.ingestion.cleaner import tokenize_words


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Computes token Jaccard similarity between two texts."""
    tokens_a = set(tokenize_words(text_a))
    tokens_b = set(tokenize_words(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


def maximal_marginal_relevance(
    query_vector: Optional[np.ndarray],
    candidates: Sequence[ScoredChunk],
    candidate_embeddings: Optional[np.ndarray] = None,
    top_k: int = 5,
    lambda_mult: float = 0.7
) -> List[ScoredChunk]:
    """
    Selects top_k chunks using Maximal Marginal Relevance:
        MMR = argmax_{d in R \\ S} [ lambda * Sim1(d, Q) - (1 - lambda) * max_{s in S} Sim2(d, s) ]
    
    Args:
        query_vector: Dense embedding of the query (optional if candidate scores used for Sim1).
        candidates: Sequence of ScoredChunks retrieved from prior stage.
        candidate_embeddings: 2D numpy matrix of shape (len(candidates), D) matching candidates.
        top_k: Number of chunks to select.
        lambda_mult: Trade-off parameter in [0.0, 1.0].
                     1.0 = Pure relevance (greedy ranking).
                     0.0 = Pure diversity (maximum novelty).
                     0.7 = Balanced relevance and diversity (recommended).
                     
    Returns:
        List of selected ScoredChunks in order of selection.
    """
    if not (0.0 <= lambda_mult <= 1.0):
        raise ValueError(f"lambda_mult must be in [0.0, 1.0], got {lambda_mult}")

    if not candidates:
        return []

    n_candidates = len(candidates)
    if top_k >= n_candidates:
        return list(candidates)

    # 1. Compute Sim1(d, Q) for all candidates
    sim1_scores = np.zeros(n_candidates, dtype=np.float64)
    if query_vector is not None and candidate_embeddings is not None and candidate_embeddings.shape[0] == n_candidates:
        sim1_scores = batch_cosine_similarity(query_vector, candidate_embeddings)
    else:
        # Use existing candidate scores
        scores = [sc.score for sc in candidates]
        min_s = min(scores)
        max_s = max(scores)
        if min_s >= 0.0 and max_s <= 1.0:
            sim1_scores = np.array(scores, dtype=np.float64)
        else:
            diff = max_s - min_s
            if diff == 0:
                sim1_scores = np.ones(n_candidates, dtype=np.float64)
            else:
                sim1_scores = np.array([(s - min_s) / diff for s in scores], dtype=np.float64)

    # 2. Iterative selection
    selected_indices: List[int] = []
    remaining_indices = list(range(n_candidates))

    # Pick the chunk with highest relevance first
    first_choice = int(np.argmax(sim1_scores))
    selected_indices.append(first_choice)
    remaining_indices.remove(first_choice)

    while len(selected_indices) < top_k and remaining_indices:
        best_mmr_score = -float("inf")
        best_candidate_idx = -1

        for idx in remaining_indices:
            rel = sim1_scores[idx]

            # Compute max similarity to already selected chunks
            max_sim_to_selected = 0.0
            if candidate_embeddings is not None and candidate_embeddings.shape[0] == n_candidates:
                selected_embs = candidate_embeddings[selected_indices]
                curr_emb = candidate_embeddings[idx]
                sims = batch_cosine_similarity(curr_emb, selected_embs)
                max_sim_to_selected = float(np.max(sims)) if len(sims) > 0 else 0.0
            else:
                # Fallback to Jaccard similarity on chunk text
                text_idx = candidates[idx].chunk.text
                max_sim_to_selected = max(
                    jaccard_similarity(text_idx, candidates[s_idx].chunk.text)
                    for s_idx in selected_indices
                )

            # MMR equation
            mmr_score = lambda_mult * rel - (1.0 - lambda_mult) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate_idx = idx

        if best_candidate_idx == -1:
            break

        selected_indices.append(best_candidate_idx)
        remaining_indices.remove(best_candidate_idx)

    return [candidates[idx] for idx in selected_indices]


class MaximalMarginalRelevance:
    """Stateful wrapper for MMR."""
    def __init__(self, lambda_mult: float = 0.7):
        self.lambda_mult = lambda_mult

    def select(
        self,
        candidates: Sequence[ScoredChunk],
        query_vector: Optional[np.ndarray] = None,
        candidate_embeddings: Optional[np.ndarray] = None,
        top_k: int = 5,
        lambda_mult: Optional[float] = None
    ) -> List[ScoredChunk]:
        """Selects diverse chunks using MMR."""
        lam = self.lambda_mult if lambda_mult is None else lambda_mult
        return maximal_marginal_relevance(
            query_vector=query_vector,
            candidates=candidates,
            candidate_embeddings=candidate_embeddings,
            top_k=top_k,
            lambda_mult=lam
        )
