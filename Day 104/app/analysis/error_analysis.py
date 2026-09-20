"""
Error analysis module for Day 104 Semantic Search Engine.
Identifies false positives, false negatives, and categorizes failure modes.
"""

from typing import Any, Dict, List, Set
import pandas as pd


class ErrorAnalyzer:
    """Performs qualitative and quantitative error diagnosis on search results."""

    @staticmethod
    def analyze_query_errors(
        engine: Any,
        queries: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> pd.DataFrame:
        """Identify false positives, false negatives, and failure reasons for queries.
        
        Args:
            engine: Search engine instance.
            queries: Benchmark queries list.
            documents: Document collection list.
            top_k: Top-K retrieval cut-off.
            
        Returns:
            DataFrame detailing error cases per query.
        """
        doc_map = {doc["id"]: doc for doc in documents}
        error_rows: List[Dict[str, Any]] = []

        for q in queries:
            qid = q.get("query_id")
            query_str = q.get("query", "")
            relevant_ids: Set[Any] = set(q.get("relevant_doc_ids", []))
            query_type = q.get("query_type", "general")

            results = engine.search(query_str, top_k=top_k)
            retrieved_ids = [r["document_id"] for r in results]

            false_positives = [doc_id for doc_id in retrieved_ids if doc_id not in relevant_ids]
            false_negatives = [doc_id for doc_id in relevant_ids if doc_id not in retrieved_ids]

            # Diagnose failure mode
            failure_mode = "None"
            if false_negatives and not false_positives:
                failure_mode = "Low Recall (Under-retrieval)"
            elif false_positives and not false_negatives:
                failure_mode = "Low Precision (Surplus noise)"
            elif false_positives and false_negatives:
                if query_type == "word_order":
                    failure_mode = "Word-Order Insensitivity (Mean Pooling)"
                elif query_type == "semantic_synonym":
                    failure_mode = "Vocabulary Mismatch / Synonymy Gap"
                elif query_type == "short_query":
                    failure_mode = "High Ambiguity (Short Query)"
                else:
                    failure_mode = "Semantic Drift / Partial Overlap"

            fp_titles = [doc_map[did]["title"] for did in false_positives if did in doc_map]
            fn_titles = [doc_map[did]["title"] for did in false_negatives if did in doc_map]

            error_rows.append({
                "query_id": qid,
                "query": query_str,
                "query_type": query_type,
                "num_relevant": len(relevant_ids),
                "num_retrieved": len(retrieved_ids),
                "num_false_positives": len(false_positives),
                "num_false_negatives": len(false_negatives),
                "failure_mode": failure_mode,
                "false_positive_titles": "; ".join(fp_titles[:3]),
                "false_negative_titles": "; ".join(fn_titles[:3])
            })

        return pd.DataFrame(error_rows)
