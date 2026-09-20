"""
Main entry point for Day 104: Semantic Search & Document Embeddings.
Executes the full pipeline: data validation, indexing, evaluation, error analysis,
visualization, and report generation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add app parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import config
from app.data.loader import DataLoader
from app.data.validator import DataValidator
from app.data.ground_truth import GroundTruthManager
from app.preprocessing.cleaner import TextCleaner
from app.preprocessing.tokenizer import Tokenizer
from app.representations.document_embedding import DocumentEmbedder
from app.retrieval.ranking import cosine_similarity
from app.retrieval.tfidf_search import TfidfSearchEngine
from app.retrieval.embedding_search import EmbeddingSearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.evaluation.evaluator import SearchEvaluator
from app.analysis.error_analysis import ErrorAnalyzer
from app.analysis.similarity_analysis import SimilarityAnalyzer
from app.visualization.charts import SearchVisualizer
from app.report import ReportGenerator


# Ensure safe utf-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_pipeline():
    print("=" * 70)
    print("[DAY 104] SEMANTIC SEARCH & DOCUMENT EMBEDDINGS")
    print("=" * 70)

    config.ensure_directories()

    # 1. Load data
    print("\n[1/7] Loading and validating dataset...")
    docs = DataLoader.load_documents(config.DOCUMENTS_PATH)
    queries = DataLoader.load_ground_truth(config.GROUND_TRUTH_PATH)

    valid_docs, doc_errors = DataValidator.validate_documents(docs)
    if not valid_docs:
        print(f"[ERROR] Document validation failed: {doc_errors}")
        sys.exit(1)

    doc_ids = {d["id"] for d in docs}
    valid_queries, query_errors = DataValidator.validate_ground_truth(queries, valid_doc_ids=doc_ids)
    if not valid_queries:
        print(f"[ERROR] Ground truth validation failed: {query_errors}")
        sys.exit(1)

    print(f"[OK] Loaded {len(docs)} documents and {len(queries)} benchmark queries.")

    # 2. Build engines
    print("\n[2/7] Indexing Search Engines...")
    cleaner = TextCleaner()
    tokenizer = Tokenizer()

    # Engine A: TF-IDF
    tfidf_engine = TfidfSearchEngine(cleaner=cleaner)
    tfidf_engine.index(docs)
    print(f"  • TF-IDF Engine indexed with vocabulary size: {tfidf_engine.tfidf.vocab_size}")

    # Engine B: Embeddings
    embedder = DocumentEmbedder(embedding_dim=config.EMBEDDING_DIM, seed=config.RANDOM_SEED)
    embedding_engine = EmbeddingSearchEngine(
        embedder=embedder,
        cleaner=cleaner,
        tokenizer=tokenizer,
        pooling=config.DEFAULT_POOLING
    )
    embedding_engine.index(docs, tfidf_model=tfidf_engine.tfidf, train_embeddings=True)
    embedder.save_embeddings(config.EMBEDDINGS_PATH)
    print(f"  • Embedding Engine indexed with {len(embedder.embeddings)} word vectors (dim={config.EMBEDDING_DIM})")

    # Engine C: Hybrid
    hybrid_engine = HybridSearchEngine(
        tfidf_engine=tfidf_engine,
        embedding_engine=embedding_engine,
        default_alpha=config.HYBRID_ALPHA
    )
    print(f"  • Hybrid Engine initialized with alpha={config.HYBRID_ALPHA}")

    # 3. Benchmark Evaluation
    print("\n[3/7] Running Benchmark Evaluation across all engines...")
    evaluator = SearchEvaluator(k_values=config.EVAL_K_VALUES)

    tfidf_df, tfidf_summary = evaluator.evaluate_engine(tfidf_engine, queries, top_k=config.DEFAULT_TOP_K)
    emb_df, emb_summary = evaluator.evaluate_engine(embedding_engine, queries, top_k=config.DEFAULT_TOP_K)
    hyb_df, hyb_summary = evaluator.evaluate_engine(hybrid_engine, queries, top_k=config.DEFAULT_TOP_K)

    metrics_dict = {
        "TF-IDF": tfidf_summary,
        "Embedding": emb_summary,
        "Hybrid": hyb_summary
    }

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} | {'TF-IDF':<11} | {'Embedding':<11} | {'Hybrid':<11}")
    print("-" * 65)
    for m in ["Mean Precision@1", "Mean Precision@3", "Mean Precision@5", "Mean Recall@5", "MRR"]:
        print(f"{m:<25} | {tfidf_summary[m]:<11.4f} | {emb_summary[m]:<11.4f} | {hyb_summary[m]:<11.4f}")
    print("=" * 65)

    # 4. Error Analysis
    print("\n[4/7] Performing Error Analysis...")
    error_df = ErrorAnalyzer.analyze_query_errors(
        engine=embedding_engine,
        queries=queries,
        documents=docs,
        top_k=config.DEFAULT_TOP_K
    )
    error_df.to_csv(config.ERRORS_PATH, index=False)
    print(f"  • Error analysis saved to {config.ERRORS_PATH}")

    # 5. Save Search Results and Metrics
    print("\n[5/7] Saving Results & Metrics CSVs...")
    # Compile detailed search results for all queries
    search_records = []
    for q in queries:
        qid = q["query_id"]
        q_text = q["query"]
        hyb_res = hybrid_engine.search(q_text, top_k=config.DEFAULT_TOP_K)
        for r in hyb_res:
            search_records.append({
                "query_id": qid,
                "query": q_text,
                "rank": r["rank"],
                "document_id": r["document_id"],
                "title": r["title"],
                "category": r["category"],
                "hybrid_score": r["hybrid_score"],
                "tfidf_score": r["tfidf_score"],
                "embedding_score": r["embedding_score"],
                "is_relevant": r["document_id"] in q["relevant_doc_ids"]
            })

    results_df = pd.DataFrame(search_records)
    results_df.to_csv(config.SEARCH_RESULTS_PATH, index=False)

    # Save summary metrics
    metrics_rows = []
    for eng_name, metrics in metrics_dict.items():
        row = {"engine": eng_name}
        row.update(metrics)
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(config.RETRIEVAL_METRICS_PATH, index=False)
    print(f"  • Search results saved to {config.SEARCH_RESULTS_PATH}")
    print(f"  • Metrics summary saved to {config.RETRIEVAL_METRICS_PATH}")

    # 6. Generate 12 Visualizations
    print("\n[6/7] Generating Visualizations (12 charts)...")
    visualizer = SearchVisualizer(config.CHARTS_DIR)

    # Collect similarity score matrices for distributions
    all_tfidf_scores = np.array([tfidf_engine.get_scores(q["query"]) for q in queries])
    all_emb_scores = np.array([embedding_engine.get_scores(q["query"]) for q in queries])

    # 1. Category distribution
    visualizer.plot_category_distribution(docs)
    # 2. Document length distribution
    visualizer.plot_document_length_distribution(docs)
    # 3. Query length distribution
    visualizer.plot_query_length_distribution(queries)
    # 4. Score distribution
    visualizer.plot_similarity_score_distribution(all_tfidf_scores, all_emb_scores)
    # 5. Precision comparison
    visualizer.plot_precision_comparison(metrics_dict)
    # 6. Recall comparison
    visualizer.plot_recall_comparison(metrics_dict)
    # 7. MRR comparison
    visualizer.plot_mrr_comparison(metrics_dict)
    # 8. Score correlation
    visualizer.plot_tfidf_vs_embedding_correlation(all_tfidf_scores, all_emb_scores)

    # 9. Top retrieved scores by rank
    top_scores_records = []
    for q in queries[:10]:
        t_res = tfidf_engine.search(q["query"], top_k=5)
        e_res = embedding_engine.search(q["query"], top_k=5)
        for r in t_res:
            top_scores_records.append({"rank": r["rank"], "score": r["score"], "engine": "TF-IDF"})
        for r in e_res:
            top_scores_records.append({"rank": r["rank"], "score": r["score"], "engine": "Embedding"})
    top_scores_df = pd.DataFrame(top_scores_records)
    visualizer.plot_top_retrieved_scores(top_scores_df)

    # 10. Error analysis breakdown
    visualizer.plot_error_analysis(error_df)

    # 11. PCA 2D embedding plot
    visualizer.plot_embedding_pca(embedding_engine.doc_matrix, docs)

    # 12. Query-document similarity heatmap for a subset of queries and documents
    sample_queries = queries[:6]
    sample_docs = docs[:8]
    heatmap_mat = np.zeros((len(sample_queries), len(sample_docs)))
    for i, q in enumerate(sample_queries):
        for j, d in enumerate(sample_docs):
            heatmap_mat[i, j] = cosine_similarity(
                embedding_engine.embedder.embed_document(tokenizer.tokenize(cleaner.clean(q["query"]))),
                embedding_engine.doc_matrix[j]
            )
    q_labels = [q["query_id"] + ": " + q["query"][:20] for q in sample_queries]
    d_labels = [f"D{d['id']}: " + d["title"][:15] for d in sample_docs]
    visualizer.plot_query_doc_heatmap(heatmap_mat, q_labels, d_labels)

    print(f"  • Successfully generated all 12 charts in {config.CHARTS_DIR}")

    # 7. Generate Final Markdown Report
    print("\n[7/7] Generating Markdown Report...")
    ReportGenerator.generate_report(metrics_dict, tfidf_df, error_df, config.REPORT_PATH)
    print(f"  • Report generated at {config.REPORT_PATH}")

    print("\n[SUCCESS] Day 104 Semantic Search Pipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
