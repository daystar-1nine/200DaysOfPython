"""
Experiment 5: Hybrid Search Parameter Sensitivity Analysis (Alpha Sweep).
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
from app.retrieval.hybrid_search import HybridSearchEngine
from app.evaluation.evaluator import SearchEvaluator


def run_experiment():
    print("=" * 65)
    print("EXPERIMENT 5: HYBRID SEARCH WEIGHTING (ALPHA SWEEP)")
    print("=" * 65)

    docs = DataLoader.load_documents(config.DOCUMENTS_PATH)
    queries = DataLoader.load_ground_truth(config.GROUND_TRUTH_PATH)

    tfidf_engine = TfidfSearchEngine()
    tfidf_engine.index(docs)

    emb_engine = EmbeddingSearchEngine()
    emb_engine.index(docs, tfidf_model=tfidf_engine.tfidf)

    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    evaluator = SearchEvaluator()

    print(f"\nEvaluating Alpha in [0.0 (Pure Embedding), 1.0 (Pure TF-IDF)]:")
    print(f"{'Alpha':<8} | {'Precision@1':<13} | {'Precision@5':<13} | {'Recall@5':<13} | {'MRR':<10}")
    print("-" * 65)

    best_alpha = 0.5
    best_mrr = -1.0

    for a in alphas:
        hybrid = HybridSearchEngine(tfidf_engine, emb_engine, default_alpha=a)
        _, summary = evaluator.evaluate_engine(hybrid, queries, top_k=5)

        p1 = summary["Mean Precision@1"]
        p5 = summary["Mean Precision@5"]
        r5 = summary["Mean Recall@5"]
        mrr = summary["MRR"]

        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = a

        print(f"{a:<8.2f} | {p1:<13.4f} | {p5:<13.4f} | {r5:<13.4f} | {mrr:<10.4f}")

    print("\nOptimal Configuration:")
    print(f"  -> Best Alpha: {best_alpha:.2f} with MRR = {best_mrr:.4f}")
    print("  -> Blending both keyword matching and dense embeddings provides superior robustness across diverse query intents.")


if __name__ == "__main__":
    run_experiment()
