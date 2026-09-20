"""Experiment 2: Context Window Size Comparison (1, 2, 3)."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data.corpus import get_corpus
from app.data.tokenizer import tokenize_corpus
from app.embeddings.word2vec import Word2Vec
from app.preprocessing.context_pairs import generate_all_pairs
from app.data.vocabulary import Vocabulary

def run():
    print("=== EXPERIMENT 2: CONTEXT WINDOW SIZES (1, 2, 3) ===")
    corpus = tokenize_corpus(get_corpus())
    vocab = Vocabulary().build_vocab(corpus)
    windows = [1, 2, 3]
    results = []
    
    for w in windows:
        pairs = generate_all_pairs(corpus, vocab, window_size=w)
        model = Word2Vec(window_size=w, epochs=30, seed=42)
        model.fit(corpus)
        
        results.append({
            "window_size": w,
            "total_pairs": len(pairs),
            "final_loss": model.epoch_losses[-1],
            "cat_top_neighbor": model.most_similar("cat", top_k=1)[0][0] if model.most_similar("cat") else "N/A"
        })
        print(f"Window {w} | Total Pairs: {len(pairs)} | Final Loss: {model.epoch_losses[-1]:.4f}")
        
    df = pd.DataFrame(results)
    print("\nSummary:\n", df.to_string())
    return df

if __name__ == "__main__":
    run()
