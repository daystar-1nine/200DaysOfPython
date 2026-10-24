"""
CLI entry point for the Day 117 Mini RAG Document Question Answering System.
"""
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
day117_dir = current_dir.parent
if str(day117_dir) not in sys.path:
    sys.path.insert(0, str(day117_dir))

from app.config import RAGConfig
from app.pipeline import RAGPipeline
from app.evaluation.evaluator import RAGEvaluator


def cmd_ingest(args):
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    data_dir = Path(args.data) if args.data else config.data_raw_dir

    print("=" * 70)
    print("  [Mini RAG] Ingesting Documents from:", data_dir)
    print("=" * 70)

    count = pipeline.ingest_directory(data_dir)
    print(f"Successfully processed and indexed {count} document chunks.")

    index_dir = Path(args.index) if args.index else config.index_dir
    print(f"Saving vector and lexical index to: {index_dir}")
    pipeline.save_index(index_dir)
    print("Indexing complete!")


def cmd_query(args):
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    index_dir = Path(args.index) if args.index else config.index_dir

    if not (index_dir / "metadata.json").exists():
        print(f"Index not found at {index_dir}. Please run 'ingest' first.")
        sys.exit(1)

    pipeline.load_index(index_dir)

    print("=" * 70)
    print(f"  [Mini RAG Query] Method: {args.method.upper()} | Top-K: {args.top_k}")
    print("=" * 70)
    print(f"Question: {args.question}\n")

    response = pipeline.query(
        question=args.question,
        top_k=args.top_k,
        method=args.method
    )

    print("-" * 70)
    print(f"Answer:\n{response.answer}\n")
    print("-" * 70)
    print(f"Supported Grounding: {response.is_supported}")
    print(f"Latency: {response.generation_time_ms:.2f} ms")
    if response.citations:
        print(f"Attributed Citations ({len(response.citations)}):")
        for c in response.citations:
            print(f"  * {c.format_citation_tag()} (Score: {c.relevance_score:.4f})")


def cmd_evaluate(args):
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    index_dir = Path(args.index) if args.index else config.index_dir

    if not (index_dir / "metadata.json").exists():
        print(f"Index not found at {index_dir}. Ingesting raw documents first...")
        pipeline.ingest_directory(config.data_raw_dir)
        pipeline.save_index(index_dir)
    else:
        pipeline.load_index(index_dir)

    eval_file = Path(args.eval_file) if args.eval_file else config.data_eval_dir / "eval_questions.json"
    if not eval_file.exists():
        print(f"Evaluation dataset not found at {eval_file}")
        sys.exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluator = RAGEvaluator(pipeline)
    print(f"Evaluating {len(data)} benchmark questions using method='{args.method}' (Top-K={args.top_k})...")
    res = evaluator.evaluate_benchmark(data, top_k=args.top_k, method=args.method)

    summary = res["summary"]
    print("\n" + "=" * 70)
    print("  RAG BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:<28}: {v}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Day 117 Mini RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents and build vector index")
    p_ingest.add_argument("--data", type=str, help="Directory containing raw documents")
    p_ingest.add_argument("--index", type=str, help="Output directory for index")

    # Query
    p_query = subparsers.add_parser("query", help="Ask a question against indexed documents")
    p_query.add_argument("question", type=str, help="User query text")
    p_query.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    p_query.add_argument("--method", type=str, default="hybrid", choices=["keyword", "semantic", "hybrid"], help="Retrieval method")
    p_query.add_argument("--index", type=str, help="Directory containing index")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate retrieval and generation metrics")
    p_eval.add_argument("--eval-file", type=str, help="Path to evaluation questions JSON")
    p_eval.add_argument("--top-k", type=int, default=5, help="Top-K for evaluation")
    p_eval.add_argument("--method", type=str, default="hybrid", help="Retrieval method")
    p_eval.add_argument("--index", type=str, help="Directory containing index")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
