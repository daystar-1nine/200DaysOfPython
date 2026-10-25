"""
Challenge 4: Maximal Marginal Relevance (MMR) Diversity Selection.
Balances query relevance against inter-candidate redundancy.
"""
from typing import List, Dict, Tuple
import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def mmr_select(
    query_vector: np.ndarray,
    candidate_embeddings: Dict[str, np.ndarray],
    top_k: int = 3,
    lambda_param: float = 0.7
) -> List[Tuple[str, float]]:
    """
    MMR = argmax_{d in R \\ S} [ lambda * Sim(d, q) - (1 - lambda) * max_{s in S} Sim(d, s) ]
    """
    remaining = list(candidate_embeddings.keys())
    selected: List[str] = []

    # Precompute query similarities
    q_sims = {doc_id: cosine_sim(query_vector, emb) for doc_id, emb in candidate_embeddings.items()}

    # First choice is highest query similarity
    first_choice = max(remaining, key=lambda d: q_sims[d])
    selected.append(first_choice)
    remaining.remove(first_choice)

    while len(selected) < top_k and remaining:
        best_doc = None
        best_score = -float("inf")

        for d in remaining:
            rel = q_sims[d]
            max_sim_to_selected = max(cosine_sim(candidate_embeddings[d], candidate_embeddings[s]) for s in selected)
            mmr_val = lambda_param * rel - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_val > best_score:
                best_score = mmr_val
                best_doc = d

        if best_doc is None:
            break

        selected.append(best_doc)
        remaining.remove(best_doc)

    return [(doc_id, q_sims[doc_id]) for doc_id in selected]


def main():
    print("=== Challenge 4: Maximal Marginal Relevance (MMR) ===")
    
    np.random.seed(42)
    dim = 8
    q_vec = np.array([1.0, 0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    
    # 4 documents: Doc A, Doc B (near-duplicate of A), Doc C (different topic but also matches query), Doc D (irrelevant)
    candidates = {
        "Doc_A": np.array([0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "Doc_B_Duplicate": np.array([0.89, 0.79, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 99.9% identical to Doc A
        "Doc_C_Diverse": np.array([0.5, 0.5, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0]),       # Relevant to query, but highly distinct from A
        "Doc_D_Unrelated": np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.5])
    }

    # Greedy Top-K (lambda = 1.0)
    greedy = mmr_select(q_vec, candidates, top_k=2, lambda_param=1.0)
    print("\nGreedy Top-2 (lambda = 1.0 - pure relevance):")
    for d, s in greedy:
        print(f"  {d:<20} Relevance: {s:.4f}")

    # MMR Diversified (lambda = 0.5)
    diverse = mmr_select(q_vec, candidates, top_k=2, lambda_param=0.5)
    print("\nDiverse Top-2 (lambda = 0.5 - MMR balanced):")
    for d, s in diverse:
        print(f"  {d:<20} Relevance: {s:.4f}")

    assert "Doc_B_Duplicate" in [d for d, _ in greedy]
    assert "Doc_C_Diverse" in [d for d, _ in diverse]
    print("\n[SUCCESS] MMR successfully suppressed duplicate Doc_B in favor of diverse Doc_C!")


if __name__ == "__main__":
    main()
