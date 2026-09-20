"""Experiment 1: Embedding Dimension Comparison (8, 16, 32, 64)."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data.corpus import get_corpus
from app.data.tokenizer import tokenize_corpus
from app.embeddings.word2vec import Word2Vec

def run():
    print("=== EXPERIMENT 1: EMBEDDING DIMENSIONS (8, 16, 32, 64) ===")
    corpus = tokenize_corpus(get_corpus())
    dims = [8, 16, 32, 64]
    results = []
    
    for d in dims:
        t0 = time.perf_counter()
        model = Word2Vec(embedding_dim=d, epochs=30, seed=42)
        model.fit(corpus)
        elapsed = (time.perf_counter() - t0) * 1000
        
        results.append({
            "dimension": d,
            "initial_loss": model.epoch_losses[0],
            "final_loss": model.epoch_losses[-1],
            "train_time_ms": elapsed,
            "cat_top_neighbor": model.most_similar("cat", top_k=1)[0][0] if model.most_similar("cat") else "N/A"
        })
        print(f"Dim {d:2d} | Init Loss: {model.epoch_losses[0]:.4f} | Final Loss: {model.epoch_losses[-1]:.4f} | Time: {elapsed:.1f}ms")
        
    df = pd.DataFrame(results)
    print("\nSummary:\n", df.to_string())
    return df

if __name__ == "__main__":
    run()
