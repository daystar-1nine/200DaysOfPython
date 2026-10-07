"""
Multi-model comparative benchmark engine.
Runs Base vs SFT vs DPO evaluation, pairwise win rates, bootstrap CIs, and metrics matrix exports.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
import pandas as pd
import numpy as np

from app.datasets.schema import EvaluationExample
from app.generation.generator import ModelResponseGenerator
from app.generation.settings import EvaluationSettings
from app.evaluators import (
    normalized_exact_match,
    sentence_bleu,
    compute_rouge_all,
    semantic_similarity,
    evaluate_instruction_following,
    evaluate_factuality_example,
    evaluate_coding_example,
    evaluate_safety_example,
    pairwise_win_rate,
    compare_two_responses
)
from app.analysis.statistics import (
    bootstrap_confidence_interval,
    paired_permutation_test,
    paired_t_test
)
from app.analysis.error_analysis import generate_error_analysis_report


class ModelComparator:
    """
    Coordinates side-by-side benchmarking of language model variants.
    """
    def __init__(
        self,
        models: Optional[List[str]] = None,
        settings: Optional[EvaluationSettings] = None
    ):
        self.model_names = models or ["base", "sft", "dpo"]
        self.settings = settings or EvaluationSettings()
        self.generators = {
            m: ModelResponseGenerator(model_type=m, settings=self.settings)
            for m in self.model_names
        }

    def evaluate_all(
        self,
        examples: List[EvaluationExample]
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation for all models across all benchmark examples.
        Returns full results dictionary containing records, metrics, win rates, and category matrices.
        """
        all_records: List[Dict[str, Any]] = []
        per_model_scores: Dict[str, Dict[str, List[float]]] = {
            m: {
                "exact_match": [],
                "bleu": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "semantic": [],
                "instruction": [],
                "factuality": [],
                "safety": [],
                "coding": []
            }
            for m in self.model_names
        }

        # Store predictions per example for pairwise comparisons
        example_predictions: Dict[str, Dict[str, str]] = {ex.id: {} for ex in examples}

        for ex in examples:
            ref_primary = ex.reference or (ex.references[0] if ex.references else "")
            refs_all = ex.get_references()

            for m in self.model_names:
                gen = self.generators[m]
                pred = gen.generate(
                    prompt=ex.prompt,
                    context=ex.context,
                    reference=ref_primary,
                    category=ex.category
                )
                example_predictions[ex.id][m] = pred

                nem = normalized_exact_match(pred, refs_all)
                bleu_vals = sentence_bleu(pred, refs_all)
                bleu_ov = bleu_vals["bleu_overall"]
                rouge_vals = compute_rouge_all(pred, refs_all)
                sem = semantic_similarity(pred, refs_all)
                instr_score = evaluate_instruction_following(ex.prompt, pred)["score"]
                fact_score = evaluate_factuality_example(pred, ref_primary, context=ex.context, supported=ex.supported)["score"] * 100.0
                code_score = evaluate_coding_example(pred, test_cases=ex.test_cases)["score"]
                safe_score = evaluate_safety_example(ex.prompt, pred, is_sensitive=(ex.category == "safety"))["score"]

                per_model_scores[m]["exact_match"].append(nem * 100.0)
                per_model_scores[m]["bleu"].append(bleu_ov * 100.0)
                per_model_scores[m]["rouge1"].append(rouge_vals["rouge_1"] * 100.0)
                per_model_scores[m]["rouge2"].append(rouge_vals["rouge_2"] * 100.0)
                per_model_scores[m]["rougeL"].append(rouge_vals["rouge_l"] * 100.0)
                per_model_scores[m]["semantic"].append(sem * 100.0)
                per_model_scores[m]["instruction"].append(instr_score)
                per_model_scores[m]["factuality"].append(fact_score)
                per_model_scores[m]["safety"].append(safe_score)
                per_model_scores[m]["coding"].append(code_score)

                all_records.append({
                    "id": ex.id,
                    "category": ex.category,
                    "model": m,
                    "prompt": ex.prompt,
                    "response": pred,
                    "reference": ref_primary,
                    "supported": ex.supported,
                    "exact_match": nem * 100.0,
                    "bleu": round(bleu_ov * 100.0, 2),
                    "rouge1": round(rouge_vals["rouge_1"] * 100.0, 2),
                    "rouge2": round(rouge_vals["rouge_2"] * 100.0, 2),
                    "rougeL": round(rouge_vals["rouge_l"] * 100.0, 2),
                    "semantic_similarity": round(sem * 100.0, 2),
                    "instruction_score": instr_score,
                    "factuality": fact_score,
                    "safety": safe_score,
                    "coding_pass_rate": code_score
                })

        # 1. Summary comparison table
        summary_rows = []
        for m in self.model_names:
            m_scores = per_model_scores[m]
            nem_ci = bootstrap_confidence_interval(m_scores["exact_match"])
            bleu_ci = bootstrap_confidence_interval(m_scores["bleu"])
            rouge_ci = bootstrap_confidence_interval(m_scores["rougeL"])
            sem_ci = bootstrap_confidence_interval(m_scores["semantic"])
            instr_ci = bootstrap_confidence_interval(m_scores["instruction"])
            fact_ci = bootstrap_confidence_interval(m_scores["factuality"])
            safe_ci = bootstrap_confidence_interval(m_scores["safety"])
            code_ci = bootstrap_confidence_interval(m_scores["coding"])

            composite = round(
                0.20 * nem_ci["mean"] +
                0.15 * bleu_ci["mean"] +
                0.15 * rouge_ci["mean"] +
                0.15 * sem_ci["mean"] +
                0.15 * instr_ci["mean"] +
                0.10 * fact_ci["mean"] +
                0.10 * safe_ci["mean"],
                2
            )

            summary_rows.append({
                "model": m.upper(),
                "overall_score": composite,
                "exact_match": nem_ci["mean"],
                "bleu": bleu_ci["mean"],
                "rouge1": round(float(np.mean(m_scores["rouge1"])), 2),
                "rouge2": round(float(np.mean(m_scores["rouge2"])), 2),
                "rougeL": rouge_ci["mean"],
                "semantic_similarity": sem_ci["mean"],
                "instruction_score": instr_ci["mean"],
                "factuality": fact_ci["mean"],
                "safety": safe_ci["mean"],
                "coding_pass_rate": code_ci["mean"],
                "bleu_ci_95": f"[{bleu_ci['ci_lower']:.1f}, {bleu_ci['ci_upper']:.1f}]",
                "rouge_ci_95": f"[{rouge_ci['ci_lower']:.1f}, {rouge_ci['ci_upper']:.1f}]"
            })

        # 2. Pairwise Win Rates
        win_rate_results = {}
        pairs_to_test = [("base", "sft"), ("base", "dpo"), ("sft", "dpo")]
        for m_a, m_b in pairs_to_test:
            if m_a in self.model_names and m_b in self.model_names:
                wins_a, wins_b, ties = 0, 0, 0
                for ex in examples:
                    p_a = example_predictions[ex.id][m_a]
                    p_b = example_predictions[ex.id][m_b]
                    cmp = compare_two_responses(ex.prompt, p_a, p_b, reference=ex.reference)
                    if cmp["winner"] == "A":
                        wins_a += 1
                    elif cmp["winner"] == "B":
                        wins_b += 1
                    else:
                        ties += 1

                wr = pairwise_win_rate(wins_a, wins_b, ties)
                win_rate_results[f"{m_a}_vs_{m_b}"] = {
                    "model_a": m_a.upper(),
                    "model_b": m_b.upper(),
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "win_rate_a": wr["win_rate_a"],
                    "win_rate_b": wr["win_rate_b"]
                }

        # 3. Category Matrix
        cat_df = pd.DataFrame(all_records)
        cat_matrix = cat_df.groupby(["category", "model"])[["instruction_score", "factuality", "semantic_similarity"]].mean().round(2)

        return {
            "summary_table": summary_rows,
            "all_records": all_records,
            "win_rates": win_rate_results,
            "category_df": cat_df,
            "per_model_scores": per_model_scores
        }

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Saves CSV and JSONL ledgers to output directory."""
        output_dir = Path(output_dir)
        metrics_dir = output_dir / "metrics"
        raw_dir = output_dir / "raw"
        reports_dir = output_dir / "reports"

        metrics_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. model_comparison.csv
        pd.DataFrame(results["summary_table"]).to_csv(metrics_dir / "model_comparison.csv", index=False)

        # 2. predictions.jsonl
        with open(raw_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for rec in results["all_records"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 3. win_rates.csv
        pd.DataFrame(list(results["win_rates"].values())).to_csv(metrics_dir / "win_rates.csv", index=False)

        # 4. error_analysis.csv
        generate_error_analysis_report(results["all_records"], reports_dir / "error_analysis.csv")


def compare_benchmark_results(
    examples: List[EvaluationExample],
    models: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    settings: Optional[EvaluationSettings] = None
) -> Dict[str, Any]:
    """
    Convenience function to instantiate ModelComparator, evaluate examples,
    and optionally save outputs.
    """
    comparator = ModelComparator(models=models, settings=settings)
    results = comparator.evaluate_all(examples)
    if output_dir:
        comparator.save_results(results, Path(output_dir))
    return results
