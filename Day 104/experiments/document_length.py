"""
Experiment 3: Impact of Document Length on Mean Pooling vs TF-IDF Representations.
"""

import sys
from pathlib import Path
import numpy as np
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
from app.retrieval.ranking import cosine_similarity


def run_experiment():
    print("=" * 65)
    print("EXPERIMENT 3: DOCUMENT LENGTH DILUTION EFFECT IN MEAN POOLING")
    print("=" * 65)

    docs = DataLoader.load_documents(config.DOCUMENTS_PATH)

    # Measure document lengths
    doc_lengths = [len(f"{d['title']} {d['text']}".split()) for d in docs]
    median_len = np.median(doc_lengths)

    short_docs = [d for d, l in zip(docs, doc_lengths) if l <= median_len]
    long_docs = [d for d, l in zip(docs, doc_lengths) if l > median_len]

    print(f"Total documents: {len(docs)}")
    print(f"Median word count: {median_len:.1f}")
    print(f"Short documents (<= {median_len:.0f} words): {len(short_docs)}")
    print(f"Long documents (> {median_len:.0f} words): {len(long_docs)}")

    # Train engines
    tfidf_engine = TfidfSearchEngine()
    tfidf_engine.index(docs)

    emb_engine = EmbeddingSearchEngine()
    emb_engine.index(docs, tfidf_model=tfidf_engine.tfidf)

    queries = DataLoader.load_ground_truth(config.GROUND_TRUTH_PATH)

    # Compare rank distribution of relevant documents for short vs long target documents
    doc_id_to_len_type = {
        d["id"]: ("Short" if len(f"{d['title']} {d['text']}".split()) <= median_len else "Long")
        for d in docs
    }

    t_ranks_short, t_ranks_long = [], []
    e_ranks_short, e_ranks_long = [], []

    for q in queries:
        t_res = [r["document_id"] for r in tfidf_engine.search(q["query"], top_k=20)]
        e_res = [r["document_id"] for r in emb_engine.search(q["query"], top_k=20)]

        for did in q["relevant_doc_ids"]:
            len_type = doc_id_to_len_type.get(did, "Short")
            # TF-IDF rank
            t_rank = (t_res.index(did) + 1) if did in t_res else 25
            # Embedding rank
            e_rank = (e_res.index(did) + 1) if did in e_res else 25

            if len_type == "Short":
                t_ranks_short.append(t_rank)
                e_ranks_short.append(e_rank)
            else:
                t_ranks_long.append(t_rank)
                e_ranks_long.append(e_rank)

    print("\nMean Rank of Relevant Documents (Lower is Better):")
    print(f"{'Target Doc Type':<18} | {'Count':<6} | {'TF-IDF Mean Rank':<18} | {'Embedding Mean Rank':<20}")
    print("-" * 70)
    print(f"{'Short Documents':<18} | {len(t_ranks_short):<6} | {np.mean(t_ranks_short):<18.2f} | {np.mean(e_ranks_short):<20.2f}")
    print(f"{'Long Documents':<18} | {len(t_ranks_long):<6} | {np.mean(t_ranks_long):<18.2f} | {np.mean(e_ranks_long):<20.2f}")

    print("\nObservation:")
    if np.mean(e_ranks_long) > np.mean(e_ranks_short):
        print("  -> Mean pooling exhibits mild semantic dilution on longer documents as irrelevant tokens accumulate in the average.")
    else:
        print("  -> Document length is well balanced across the controlled corpus.")


if __name__ == "__main__":
    run_experiment()
