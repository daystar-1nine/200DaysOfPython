"""
Full-suite evaluation engine for RAG retrieval and generation benchmarking.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import pandas as pd
import numpy as np

from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    hit_rate_at_k
)
from app.evaluation.grounding import evaluate_grounding
from app.citations.citation import verify_citation_coverage


class RAGEvaluator:
    """
    Evaluator orchestrating end-to-end evaluation of RAG pipelines against ground-truth benchmarks.
    """
    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def evaluate_item(
        self,
        item: Dict[str, Any],
        top_k: int = 5,
        method: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Evaluates a single question against ground truth relevant chunk IDs.
        """
        qid = item.get("id", "q_0")
        question = item["question"]
        relevant_chunks = item.get("relevant_chunks", [])
        expected_supported = item.get("supported", True)
        category = item.get("category", "general")
        query_type = item.get("query_type", "exact")

        start_time = time.perf_counter()
        response = self.pipeline.query(
            question=question,
            top_k=top_k,
            method=method
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        top_score = round(float(response.retrieved_chunks[0].score), 4) if response.retrieved_chunks else 0.0

        retrieved_cids = [r.chunk.metadata.chunk_id for r in response.retrieved_chunks]

        # Build effective relevant targets (chunk IDs and document IDs)
        target_set = set(relevant_chunks) | set(item.get("relevant_docs", []))
        eval_relevant = list(target_set) if target_set else []

        # Map retrieved chunks to target IDs when document-level match occurs
        eval_retrieved = []
        for r in response.retrieved_chunks:
            cid = r.chunk.metadata.chunk_id
            did = r.chunk.metadata.document_id
            if cid in target_set:
                eval_retrieved.append(cid)
            elif did in target_set:
                eval_retrieved.append(did)
            else:
                eval_retrieved.append(cid)

        # 1. Retrieval Metrics
        p1 = precision_at_k(eval_retrieved, eval_relevant, k=1)
        p3 = precision_at_k(eval_retrieved, eval_relevant, k=min(3, top_k))
        p5 = precision_at_k(eval_retrieved, eval_relevant, k=min(5, top_k))
        r1 = recall_at_k(eval_retrieved, eval_relevant, k=1)
        r3 = recall_at_k(eval_retrieved, eval_relevant, k=min(3, top_k))
        r5 = recall_at_k(eval_retrieved, eval_relevant, k=min(5, top_k))
        rr = reciprocal_rank(eval_retrieved, eval_relevant)
        hit5 = hit_rate_at_k(eval_retrieved, eval_relevant, k=min(5, top_k))

        # 2. Grounding & Faithfulness
        grounding_res = evaluate_grounding(
            question=question,
            answer=response.answer,
            context=response.context_used,
            is_supported=expected_supported
        )

        # 3. Citation Verification
        citation_res = verify_citation_coverage(response.answer, response.citations)

        # Determine error type if any
        error_type = "None"
        if not expected_supported and grounding_res["is_hallucinated"]:
            error_type = "Hallucination"
        elif expected_supported and hit5 == 0.0:
            error_type = "Wrong retrieval"
        elif expected_supported and r5 < 1.0 and r5 > 0.0:
            error_type = "Missing context"
        elif grounding_res["context_relevance"] < 0.20 and expected_supported:
            error_type = "Irrelevant context"
        elif grounding_res["faithfulness"] < 0.50 and not grounding_res["is_hallucinated"]:
            error_type = "Incomplete answer"

        return {
            "question_id": qid,
            "category": category,
            "query_type": query_type,
            "question": question,
            "expected_chunks": relevant_chunks,
            "retrieved_chunks": retrieved_cids,
            "expected_supported": expected_supported,
            "is_supported_prediction": response.is_supported,
            "top_score": top_score,
            "citations": [c.to_dict() for c in response.citations],
            "answer": response.answer,
            "precision_at_1": round(p1, 4),
            "precision_at_3": round(p3, 4),
            "precision_at_5": round(p5, 4),
            "recall_at_1": round(r1, 4),
            "recall_at_3": round(r3, 4),
            "recall_at_5": round(r5, 4),
            "reciprocal_rank": round(rr, 4),
            "hit_rate_at_5": round(hit5, 4),
            "faithfulness": grounding_res["faithfulness"],
            "context_relevance": grounding_res["context_relevance"],
            "is_hallucinated": grounding_res["is_hallucinated"],
            "citation_coverage": citation_res["coverage_rate"],
            "latency_ms": round(elapsed_ms, 2),
            "error_type": error_type
        }

    def evaluate_benchmark(
        self,
        dataset: List[Dict[str, Any]],
        top_k: int = 5,
        method: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Runs complete evaluation over all items, returning aggregated metrics and individual records.
        """
        records = [
            self.evaluate_item(item, top_k=top_k, method=method)
            for item in dataset
        ]

        df = pd.DataFrame(records)

        summary = {
            "total_questions": len(dataset),
            "retrieval_method": method,
            "top_k": top_k,
            "mean_precision_at_1": round(float(df["precision_at_1"].mean()), 4),
            "mean_precision_at_3": round(float(df["precision_at_3"].mean()), 4),
            "mean_precision_at_5": round(float(df["precision_at_5"].mean()), 4),
            "mean_recall_at_1": round(float(df["recall_at_1"].mean()), 4),
            "mean_recall_at_3": round(float(df["recall_at_3"].mean()), 4),
            "mean_recall_at_5": round(float(df["recall_at_5"].mean()), 4),
            "mrr": round(float(df["reciprocal_rank"].mean()), 4),
            "hit_rate_at_5": round(float(df["hit_rate_at_5"].mean()), 4),
            "mean_faithfulness": round(float(df["faithfulness"].mean()), 4),
            "mean_context_relevance": round(float(df["context_relevance"].mean()), 4),
            "hallucination_rate": round(float(df["is_hallucinated"].mean()), 4),
            "mean_citation_coverage": round(float(df["citation_coverage"].mean()), 4),
            "mean_latency_ms": round(float(df["latency_ms"].mean()), 2)
        }

        # Category breakdown
        category_breakdown = df.groupby("category")[
            ["precision_at_5", "recall_at_5", "reciprocal_rank", "faithfulness", "is_hallucinated"]
        ].mean().round(4).to_dict()

        return {
            "summary": summary,
            "records": records,
            "items": records,
            "category_breakdown": category_breakdown,
            "dataframe": df
        }
