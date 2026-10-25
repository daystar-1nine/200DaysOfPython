"""
Benchmark Evaluator Harness for Advanced RAG.
Runs standardized retrieval and generation evaluations across diverse configurations and benchmarks.
"""
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from collections import defaultdict

from app.evaluation.retrieval_metrics import evaluate_ranking
from app.evaluation.latency import LatencyTracker
from app.evaluation.error_analysis import diagnose_retrieval_error, ErrorAnalyzer, QueryErrorDiagnosis


class RAGEvaluator:
    """
    Evaluator harness running queries against configured RAG pipelines and aggregating IR metrics.
    """
    def __init__(self, benchmark_file: Path):
        self.benchmark_file = Path(benchmark_file)
        if not self.benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_file}")

        self.queries: List[Dict[str, Any]] = json.loads(self.benchmark_file.read_text(encoding="utf-8"))

    def evaluate_retriever(
        self,
        retriever_name: str,
        search_fn: Any,
        k_values: Sequence[int] = (1, 3, 5, 10)
    ) -> Dict[str, Any]:
        """
        Evaluates a search function over all benchmark queries.
        `search_fn(query: str, top_k: int) -> List[ScoredChunk]`
        """
        all_metrics: Dict[str, List[float]] = defaultdict(list)
        type_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        latency = LatencyTracker()
        query_results: List[Dict[str, Any]] = []

        for item in self.queries:
            qid = item["query_id"]
            q_text = item["query"]
            q_type = item["query_type"]
            target_docs = set(item.get("relevant_doc_ids", []))
            graded_rel = item.get("graded_relevance", {})

            # Time search
            t0 = latency.time_call(lambda: None)  # dummy to initiate
            t_start = latency.measurements[-1] if latency.measurements else 0.0

            scored_chunks = latency.time_call(search_fn, q_text, max(k_values))

            # Extract retrieved document IDs
            retrieved_doc_ids = [sc.chunk.metadata.document_id for sc in scored_chunks]

            # Compute IR metrics if target docs exist
            if target_docs:
                query_eval = evaluate_ranking(
                    retrieved_doc_ids=retrieved_doc_ids,
                    relevant_doc_ids=target_docs,
                    graded_relevance=graded_rel,
                    k_values=k_values
                )
                for m_key, val in query_eval.items():
                    all_metrics[m_key].append(val)
                    type_metrics[q_type][m_key].append(val)

            query_results.append({
                "query_id": qid,
                "query": q_text,
                "query_type": q_type,
                "target_docs": list(target_docs),
                "retrieved_docs": retrieved_doc_ids[:5],
                "retrieved_chunk_ids": [sc.chunk.metadata.chunk_id for sc in scored_chunks[:5]]
            })

        # Aggregate averages
        summary: Dict[str, float] = {}
        for m_key, vals in all_metrics.items():
            summary[f"mean_{m_key}"] = float(sum(vals) / max(1, len(vals)))

        # Breakdown by query type
        type_summary: Dict[str, Dict[str, float]] = {}
        for q_type, m_dict in type_metrics.items():
            type_summary[q_type] = {
                f"mean_{k}": float(sum(v) / max(1, len(v)))
                for k, v in m_dict.items()
            }

        latency_summary = latency.summarize()

        return {
            "retriever_name": retriever_name,
            "overall_metrics": summary,
            "by_query_type": type_summary,
            "latency": latency_summary,
            "query_results": query_results
        }
