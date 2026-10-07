"""
Command-Line Interface (CLI) for running the MiniGPT Evaluation Harness.
Usage:
  python -m app.evaluate --model sft --dataset data/evaluation.jsonl
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure Day 116 directory is in sys.path
DAY_116_ROOT = Path(__file__).resolve().parent.parent
if str(DAY_116_ROOT) not in sys.path:
    sys.path.insert(0, str(DAY_116_ROOT))

from app.datasets.loader import load_jsonl_dataset
from app.generation.generator import ModelResponseGenerator
from app.generation.settings import EvaluationSettings
from app.evaluators import (
    exact_match,
    normalized_exact_match,
    sentence_bleu,
    compute_rouge_all,
    semantic_similarity,
    evaluate_instruction_following,
    evaluate_factuality_example,
    evaluate_coding_example,
    evaluate_safety_example
)


def run_evaluation_for_model(
    model_name: str,
    examples: list,
    deterministic: bool = True
) -> Dict[str, Any]:
    """Runs evaluation for a specific model over a list of EvaluationExample items."""
    settings = EvaluationSettings(deterministic=deterministic)
    generator = ModelResponseGenerator(model_type=model_name, settings=settings)

    total_nem = 0.0
    total_bleu = 0.0
    total_rouge = 0.0
    total_semantic = 0.0
    total_instruction = 0.0
    total_factuality = 0.0
    total_coding = 0.0
    total_safety = 0.0

    count = len(examples)
    records = []

    for ex in examples:
        ref_primary = ex.reference or (ex.references[0] if ex.references else "")
        refs_all = ex.get_references()

        pred = generator.generate(
            prompt=ex.prompt,
            context=ex.context,
            reference=ref_primary,
            category=ex.category
        )

        nem = normalized_exact_match(pred, refs_all)
        bleu = sentence_bleu(pred, refs_all)["bleu_overall"]
        rouge = compute_rouge_all(pred, refs_all)["rouge_l"]
        sem = semantic_similarity(pred, refs_all)

        instr_res = evaluate_instruction_following(ex.prompt, pred)
        fact_res = evaluate_factuality_example(pred, ref_primary, context=ex.context, supported=ex.supported)
        code_res = evaluate_coding_example(pred, test_cases=ex.test_cases)
        is_safe_prompt = (ex.category == "safety")
        safe_res = evaluate_safety_example(ex.prompt, pred, is_sensitive=is_safe_prompt)

        total_nem += nem
        total_bleu += bleu
        total_rouge += rouge
        total_semantic += sem
        total_instruction += instr_res["score"]
        total_factuality += fact_res["score"]
        total_coding += code_res["score"]
        total_safety += safe_res["score"]

        records.append({
            "id": ex.id,
            "category": ex.category,
            "model": model_name,
            "prompt": ex.prompt,
            "response": pred,
            "reference": ref_primary,
            "normalized_exact_match": nem,
            "bleu": round(bleu, 4),
            "rouge_l": round(rouge, 4),
            "semantic": round(sem, 4)
        })

    n = max(1, count)
    avg_nem = round((total_nem / n) * 100.0, 2)
    avg_bleu = round((total_bleu / n) * 100.0, 2)
    avg_rouge = round((total_rouge / n) * 100.0, 2)
    avg_semantic = round((total_semantic / n) * 100.0, 2)
    avg_instr = round(total_instruction / n, 2)
    avg_fact = round((total_factuality / n) * 100.0, 2)
    avg_coding = round(total_coding / n, 2)
    avg_safety = round(total_safety / n, 2)

    # Composite overall score
    composite = round(
        0.20 * avg_nem +
        0.15 * avg_bleu +
        0.15 * avg_rouge +
        0.15 * avg_semantic +
        0.15 * avg_instr +
        0.10 * avg_fact +
        0.10 * avg_safety,
        2
    )

    summary = {
        "model": model_name.upper(),
        "total_examples": count,
        "overall_score": composite,
        "exact_match_pct": avg_nem,
        "bleu_score_pct": avg_bleu,
        "rouge_l_pct": avg_rouge,
        "semantic_similarity_pct": avg_semantic,
        "instruction_following": avg_instr,
        "factuality_score": avg_fact,
        "coding_pass_rate": avg_coding,
        "safety_score": avg_safety
    }

    return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description="MiniGPT Evaluation Harness CLI")
    parser.add_argument("--model", type=str, default="sft", choices=["base", "sft", "dpo", "all"], help="Model variant to evaluate")
    parser.add_argument("--dataset", type=str, default="data/evaluation.jsonl", help="Path to evaluation JSONL dataset")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit number of evaluation samples for fast runs")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Enforce deterministic greedy decoding")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = DAY_116_ROOT / ds_path

    print(f"\n=======================================================")
    print(f"MiniGPT Evaluation Harness (Day 116)")
    print(f"=======================================================")
    print(f"Dataset: {ds_path}")
    print(f"Model Selection: {args.model}")
    print(f"Deterministic: {args.deterministic}")

    examples = load_jsonl_dataset(ds_path)
    if args.sample_limit and args.sample_limit < len(examples):
        examples = examples[:args.sample_limit]
    print(f"Loaded {len(examples)} evaluation examples.\n")

    models_to_run = ["base", "sft", "dpo"] if args.model == "all" else [args.model]

    for m in models_to_run:
        result = run_evaluation_for_model(m, examples, deterministic=args.deterministic)
        s = result["summary"]

        print(f"-------------------------------------------------------")
        print(f"RESULTS FOR MODEL: {s['model']}")
        print(f"-------------------------------------------------------")
        print(f"  Overall Score           : {s['overall_score']:.2f} / 100")
        print(f"  Instruction Score       : {s['instruction_following']:.2f} / 100")
        print(f"  Factuality Score        : {s['factuality_score']:.2f}%")
        print(f"  Coding Pass Rate        : {s['coding_pass_rate']:.2f}%")
        print(f"  Safety Score            : {s['safety_score']:.2f}%")
        print(f"  Normalized Exact Match  : {s['exact_match_pct']:.2f}%")
        print(f"  BLEU-4 Score            : {s['bleu_score_pct']:.2f}%")
        print(f"  ROUGE-L Score           : {s['rouge_l_pct']:.2f}%")
        print(f"  Semantic Similarity     : {s['semantic_similarity_pct']:.2f}%\n")

    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
