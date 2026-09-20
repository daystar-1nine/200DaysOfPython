"""Experiment 3: Learning Rate Comparison (0.001, 0.01, 0.05)."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data.corpus import get_corpus
from app.data.tokenizer import tokenize_corpus
from app.embeddings.word2vec import Word2Vec

def run():
    print("=== EXPERIMENT 3: LEARNING RATES (0.001, 0.01, 0.05) ===")
    corpus = tokenize_corpus(get_corpus())
    lrs = [0.001, 0.01, 0.05]
    results = []
    
    for lr in lrs:
        model = Word2Vec(learning_rate=lr, epochs=30, seed=42)
        model.fit(corpus)
        
        loss_drop = model.epoch_losses[0] - model.epoch_losses[-1]
        results.append({
            "learning_rate": lr,
            "initial_loss": model.epoch_losses[0],
            "final_loss": model.epoch_losses[-1],
            "loss_drop": loss_drop
        })
        print(f"LR {lr:.3f} | Init Loss: {model.epoch_losses[0]:.4f} | Final Loss: {model.epoch_losses[-1]:.4f} | Drop: {loss_drop:.4f}")
        
    df = pd.DataFrame(results)
    print("\nSummary:\n", df.to_string())
    return df

if __name__ == "__main__":
    run()
