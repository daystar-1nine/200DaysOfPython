"""
Qualitative and quantitative benchmark evaluator for Day 115: Preference Optimization & RLHF.
Compares Base, SFT, and DPO models across Instruction Following, Helpfulness,
Conciseness, and Response Length.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
import torch

from app.models.dpo_model import MiniGPTChat
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import format_prompt, extract_assistant_response
from app.config import GenerationConfig


def score_response_quality(generated: str, ground_truth: str, instruction: str) -> Dict[str, float]:
    """
    Evaluates response quality across 3 dimensions:
      - instruction_following: does it address the query without degenerate loops?
      - helpfulness: does it include relevant concepts or keywords?
      - conciseness: is it direct and focused without excessive fluff or repetition?
    """
    gen = generated.strip().lower()
    gt = ground_truth.strip().lower()

    if not gen or len(gen) < 4:
        return {"instruction_following": 0.0, "helpfulness": 0.0, "conciseness": 0.0, "composite": 0.0}

    # Repetition loop penalty
    words = gen.split()
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return {"instruction_following": 0.0, "helpfulness": 0.0, "conciseness": 0.0, "composite": 0.0}

    # Keyword overlap
    gt_words = set(re.findall(r"\b\w{3,}\b", gt))
    stopwords = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "used", "which"}
    key_terms = gt_words - stopwords or gt_words

    matched = [w for w in key_terms if w in gen]
    overlap_ratio = len(matched) / max(1, len(key_terms))

    # Helpfulness
    if overlap_ratio >= 0.35 or len(matched) >= 2:
        helpfulness = 100.0
    elif overlap_ratio >= 0.15 or len(matched) >= 1:
        helpfulness = 60.0
    elif len(words) >= 4:
        helpfulness = 30.0
    else:
        helpfulness = 0.0

    # Instruction following
    if helpfulness >= 60.0 and len(words) >= 3:
        instruction_score = 100.0
    elif helpfulness >= 30.0:
        instruction_score = 50.0
    else:
        instruction_score = 0.0

    # Conciseness: penalizes extreme verbosity or character bloat
    if 3 <= len(words) <= 30:
        conciseness = 100.0
    elif len(words) <= 50:
        conciseness = 70.0
    else:
        conciseness = 30.0

    composite = round(0.4 * instruction_score + 0.4 * helpfulness + 0.2 * conciseness, 2)

    return {
        "instruction_following": instruction_score,
        "helpfulness": helpfulness,
        "conciseness": conciseness,
        "composite": composite
    }


def compare_models_on_prompts(
    models: Dict[str, MiniGPTChat],
    test_prompts: List[Dict[str, str]],
    tokenizer: ChatTokenizer,
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Evaluates multiple models side-by-side across a list of test prompts.
    Returns:
      (itemized_results, aggregate_metrics)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_cfg = generation_config or GenerationConfig(max_new_tokens=48, temperature=0.7, do_sample=True)

    for m in models.values():
        m.to(device).eval()

    itemized_results: List[Dict[str, Any]] = []
    model_totals = {name: {"instruction": 0.0, "helpfulness": 0.0, "conciseness": 0.0, "composite": 0.0, "length": 0.0} for name in models}

    for item in test_prompts:
        prompt = item["prompt"]
        gt = item.get("chosen", "")

        prompt_fmt = format_prompt(prompt, add_generation_prompt=True)
        prompt_ids = torch.tensor([tokenizer.encode(prompt_fmt)], dtype=torch.long, device=device)

        row: Dict[str, Any] = {"prompt": prompt, "chosen_reference": gt}

        for name, model in models.items():
            with torch.no_grad():
                out_ids = model.generate(prompt_ids, generation_config=gen_cfg, stop_token_id=tokenizer.end_id)
            raw_gen = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)
            reply = extract_assistant_response(raw_gen)

            scores = score_response_quality(reply, gt, prompt)
            word_count = len(reply.split())

            row[f"{name}_response"] = reply
            row[f"{name}_score"] = scores["composite"]
            row[f"{name}_length"] = word_count

            model_totals[name]["instruction"] += scores["instruction_following"]
            model_totals[name]["helpfulness"] += scores["helpfulness"]
            model_totals[name]["conciseness"] += scores["conciseness"]
            model_totals[name]["composite"] += scores["composite"]
            model_totals[name]["length"] += word_count

        itemized_results.append(row)

    n = max(1, len(test_prompts))
    aggregate_metrics: Dict[str, Dict[str, float]] = {}
    for name, totals in model_totals.items():
        aggregate_metrics[name] = {
            "instruction_following": round(totals["instruction"] / n, 2),
            "helpfulness": round(totals["helpfulness"] / n, 2),
            "conciseness": round(totals["conciseness"] / n, 2),
            "composite_score": round(totals["composite"] / n, 2),
            "avg_response_length": round(totals["length"] / n, 2)
        }

    return itemized_results, aggregate_metrics
