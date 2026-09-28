"""
Challenge 5: Create an attention heatmap where X-axis = token, Y-axis = attention query.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_query_token_heatmap(tokens: list, weights: np.ndarray, save_path: Path = None):
    """
    weights: (n_queries, len(tokens))
    """
    plt.figure(figsize=(max(6, len(tokens)*0.8), max(3, weights.shape[0]*0.6)))
    plt.imshow(weights, cmap="Blues", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.yticks(range(weights.shape[0]), [f"Query {i+1}" for i in range(weights.shape[0])])
    plt.xlabel("Key Tokens")
    plt.ylabel("Attention Queries")
    plt.title("Query-Token Attention Heatmap")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    tokens = ["urgent", "claim", "free", "cash", "prize"]
    weights = np.array([
        [0.4, 0.3, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.3, 0.3, 0.2]
    ])
    out = Path(__file__).parent / "demo_query_token_heatmap.png"
    plot_query_token_heatmap(tokens, weights, save_path=out)
    assert out.exists()
    print("Challenge 5 passed, saved demo heatmap to:", out)
