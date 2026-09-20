"""Task & Challenge 1: Cosine similarity implementation and verification."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def run():
    print("=== CHALLENGE 103: COSINE SIMILARITY ===")
    
    # 1. Same direction -> ~ 1.0
    a = np.array([1, 2, 3])
    b = np.array([2, 4, 6])
    sim_same = cosine_similarity(a, b)
    print(f"Same direction: {sim_same:.4f}")
    assert np.isclose(sim_same, 1.0)
    
    # 2. Orthogonal -> ~ 0.0
    a_ortho = np.array([1, 0])
    b_ortho = np.array([0, 1])
    sim_ortho = cosine_similarity(a_ortho, b_ortho)
    print(f"Orthogonal: {sim_ortho:.4f}")
    assert np.isclose(sim_ortho, 0.0)
    
    # 3. Opposite -> ~ -1.0
    a_opp = np.array([1, 0])
    b_opp = np.array([-1, 0])
    sim_opp = cosine_similarity(a_opp, b_opp)
    print(f"Opposite: {sim_opp:.4f}")
    assert np.isclose(sim_opp, -1.0)
    
    print("[SUCCESS] Cosine similarity verified across all edge cases!")

if __name__ == "__main__":
    run()
