"""
Experiment 1: TF-IDF vs Dense Embeddings Comparison across Benchmark Queries.
"""

import sys
from pathlib import Path
import pandas as pd

# Add app parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure safe utf-8 stdout on Windows
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
    print("EXPERIMENT 1: TF-IDF VS EMBEDDING SEARCH BENCHMARK")
    print("=" * 65)

    docs = DataLoader.load_documents(config.DOCUMENTS_PATH)
    queries = DataLoader.load_ground_truth(config.GROUND_TRUTH_PATH)

    tfidf_engine = TfidfSearchEngine()
    tfidf_engine.index(docs)

    emb_engine = EmbeddingSearchEngine()
    emb_engine.index(docs, tfidf_model=tfidf_engine.tfidf)

    evaluator = SearchEvaluator()
    t_df, t_summary = evaluator.evaluate_engine(tfidf_engine, queries, top_k=5)
    e_df, e_summary = evaluator.evaluate_engine(emb_engine, queries, top_k=5)

    print("\nOverall Performance Summary:")
    print(f"{'Metric':<20} | {'TF-IDF':<12} | {'Embedding':<12} | {'Delta (Emb - TFIDF)':<20}")
    print("-" * 70)
    for m in ["Mean Precision@1", "Mean Precision@3", "Mean Precision@5", "Mean Recall@5", "MRR"]:
        t_val = t_summary[m]
        e_val = e_summary[m]
        delta = e_val - t_val
        print(f"{m:<20} | {t_val:<12.4f} | {e_val:<12.4f} | {delta:+<20.4f}")

    # Query Type Breakdown
    print("\nMRR by Query Type:")
    qtypes = sorted(list(set(q["query_type"] for q in queries)))
    print(f"{'Query Type':<20} | {'Count':<6} | {'TF-IDF MRR':<12} | {'Embedding MRR':<12}")
    print("-" * 56)
    for qt in qtypes:
        t_mrr = t_df[t_df["query_type"] == qt]["reciprocal_rank"].mean()
        e_mrr = e_df[e_df["query_type"] == qt]["reciprocal_rank"].mean()
        cnt = len(t_df[t_df["query_type"] == qt])
        print(f"{qt:<20} | {cnt:<6} | {t_mrr:<12.4f} | {e_mrr:<12.4f}")


if __name__ == "__main__":
    run_experiment()
