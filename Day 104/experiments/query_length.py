"""
Experiment 2: Retrieval Performance by Query Length (Short vs Medium vs Long).
"""

import sys
from pathlib import Path
import pandas as pd

# Add app parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import config
from app.data.loader import DataLoader
from app.retrieval.tfidf_search import TfidfSearchEngine
from app.retrieval.embedding_search import EmbeddingSearchEngine
from app.evaluation.evaluator import SearchEvaluator


def run_experiment():
    print("=" * 65)
    print("EXPERIMENT 2: IMPACT OF QUERY LENGTH ON RETRIEVAL ACCURACY")
    print("=" * 65)

    docs = DataLoader.load_documents(config.DOCUMENTS_PATH)
    queries = DataLoader.load_ground_truth(config.GROUND_TRUTH_PATH)

    # Classify queries by length
    for q in queries:
        words = q["query"].split()
        if len(words) <= 2:
            q["length_bucket"] = "Short (1-2 words)"
        elif len(words) <= 5:
            q["length_bucket"] = "Medium (3-5 words)"
        else:
            q["length_bucket"] = "Long (6+ words)"

    tfidf_engine = TfidfSearchEngine()
    tfidf_engine.index(docs)

    emb_engine = EmbeddingSearchEngine()
    emb_engine.index(docs, tfidf_model=tfidf_engine.tfidf)

    evaluator = SearchEvaluator()
    t_df, _ = evaluator.evaluate_engine(tfidf_engine, queries, top_k=5)
    e_df, _ = evaluator.evaluate_engine(emb_engine, queries, top_k=5)

    buckets = ["Short (1-2 words)", "Medium (3-5 words)", "Long (6+ words)"]
    print(f"{'Query Length Bucket':<22} | {'Count':<6} | {'TF-IDF MRR':<12} | {'Embedding MRR':<14} | {'TF-IDF P@1':<12} | {'Emb P@1':<10}")
    print("-" * 85)

    for b in buckets:
        indices = [i for i, q in enumerate(queries) if q["length_bucket"] == b]
        t_sub = t_df.iloc[indices]
        e_sub = e_df.iloc[indices]

        t_mrr = t_sub["reciprocal_rank"].mean()
        e_mrr = e_sub["reciprocal_rank"].mean()
        t_p1 = t_sub["precision@1"].mean()
        e_p1 = e_sub["precision@1"].mean()

        print(f"{b:<22} | {len(indices):<6} | {t_mrr:<12.4f} | {e_mrr:<14.4f} | {t_p1:<12.4f} | {e_p1:<10.4f}")


if __name__ == "__main__":
    run_experiment()
