"""
Command-line interface for Day 118 Advanced RAG Engine.
"""
import sys
import argparse
from pathlib import Path

from app.config import AdvancedRAGConfig
from app.pipeline import AdvancedRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Day 118 Advanced RAG System")
    parser.add_argument("--query", "-q", type=str, help="Question to ask the RAG system")
    parser.add_argument("--method", "-m", type=str, default="hybrid_rrf",
                        choices=["bm25", "dense", "hybrid_weighted", "hybrid_rrf"],
                        help="Retrieval method")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranker")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR diversity")
    parser.add_argument("--no-compression", action="store_true", help="Disable context compression")
    parser.add_argument("--category", type=str, default=None, help="Optional category filter")

    args = parser.parse_args()

    config = AdvancedRAGConfig()
    pipeline = AdvancedRAGPipeline(config)
    print(f"Loading knowledge base from {config.data_raw_dir}...")
    pipeline.ingest_directory(config.data_raw_dir)
    print(f"Indexed {len(pipeline.documents)} documents into {len(pipeline.chunks)} chunks.")

    if not args.query:
        print("\nReady! Pass --query 'your question' to run a query.")
        return

    filter_fn = None
    if args.category:
        filter_fn = lambda c: c.metadata.category.lower() == args.category.lower()

    print(f"\nProcessing query: '{args.query}' (Method: {args.method})")
    response = pipeline.answer_query(
        query=args.query,
        top_k=args.top_k,
        method=args.method,
        enable_rerank=not args.no_rerank,
        enable_mmr=not args.no_mmr,
        enable_compression=not args.no_compression,
        filter_fn=filter_fn
    )

    print("\n" + "=" * 60)
    print("ANSWER:")
    print(response.answer)
    print("=" * 60)
    print(f"Supported: {response.is_supported}")
    print(f"Compression Ratio: {response.compression_ratio:.1%}")
    print(f"Latency: {response.generation_time_ms:.1f} ms")


if __name__ == "__main__":
    main()
