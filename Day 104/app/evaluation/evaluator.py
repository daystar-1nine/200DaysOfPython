"""
Search Evaluator module for Day 104 Semantic Search Engine.
Runs comprehensive evaluation suites across queries and calculates IR metrics.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

from .precision_at_k import precision_at_k
from .recall_at_k import recall_at_k
from .mrr import reciprocal_rank, mean_reciprocal_rank


class SearchEvaluator:
    """Evaluates search engine retrieval quality against ground-truth benchmarks."""

    def __init__(self, k_values: Tuple[int, ...] = (1, 3, 5)):
        self.k_values = k_values

    def evaluate_engine(
        self,
        engine: Any,
        queries: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Evaluate an indexed search engine over a list of benchmark queries.
        
        Args:
            engine: Object implementing `search(query_str, top_k=...)`.
            queries: List of query dicts with 'query_id', 'query', 'relevant_doc_ids'.
            top_k: Maximum retrieval depth.
            
        Returns:
            Tuple of:
                - DataFrame with per-query retrieval metrics.
                - Dictionary containing mean aggregated metrics.
        """
        rows: List[Dict[str, Any]] = []
        max_k = max(self.k_values + (top_k,))

        for q in queries:
            qid = q.get("query_id", "Unknown")
            query_str = q.get("query", "")
            relevant_ids = q.get("relevant_doc_ids", [])
            query_type = q.get("query_type", "general")
            category = q.get("category", "all")

            results = engine.search(query_str, top_k=max_k)
            retrieved_ids = [r["document_id"] for r in results]

            row: Dict[str, Any] = {
                "query_id": qid,
                "query": query_str,
                "query_type": query_type,
                "category": category,
                "num_relevant": len(relevant_ids),
                "num_retrieved": len(retrieved_ids)
            }

            # Precision at various K
            for k in self.k_values:
                row[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k=k)

            # Recall at top_k
            row[f"recall@{top_k}"] = recall_at_k(retrieved_ids, relevant_ids, k=top_k)

            # Reciprocal rank
            row["reciprocal_rank"] = reciprocal_rank(retrieved_ids, relevant_ids)

            rows.append(row)

        df = pd.DataFrame(rows)

        summary: Dict[str, float] = {}
        for k in self.k_values:
            summary[f"Mean Precision@{k}"] = float(df[f"precision@{k}"].mean())
        summary[f"Mean Recall@{top_k}"] = float(df[f"recall@{top_k}"].mean())
        summary["MRR"] = float(mean_reciprocal_rank(df["reciprocal_rank"].tolist()))

        return df, summary
