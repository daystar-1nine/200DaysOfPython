"""
Evaluation suite for instruction fine-tuning and chat models.
Implements the 0/1/2 rubric scoring protocol, benchmark comparison between
base and fine-tuned models, and category-level aggregation.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
from app.config import GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import format_chat, extract_assistant_response
from app.model.minigpt_chat import MiniGPTChat


def score_response(generated: str, ground_truth: str, instruction: str) -> Tuple[int, str]:
    """
    Evaluates response quality using the 3-tier instruction-following rubric:
      - 0: Failed (gibberish, excessive repetition, empty, or completely off-topic)
      - 1: Partially followed (contains key concepts/terms or partially correct direction)
      - 2: Correctly followed (fluent, relevant, contains core answer concepts, well-formatted)
    """
    gen = generated.strip().lower()
    gt = ground_truth.strip().lower()
    instr = instruction.strip().lower()

    if not gen or len(gen) < 4:
        return 0, "Failed: Response is empty or too short."

    # Repetition check (e.g. repeated words or characters)
    words = gen.split()
    if len(words) >= 6:
        repeated_word_ratio = len(set(words)) / len(words)
        if repeated_word_ratio < 0.35:
            return 0, "Failed: Degenerate word repetition."

    # Check for character repetition loops
    if re.search(r"(.)\1{6,}", gen):
        return 0, "Failed: Degenerate character repetition loop."

    # Keyword overlap analysis with ground truth
    gt_words = set(re.findall(r"\b\w{3,}\b", gt))
    # Remove common stopwords from matching
    stopwords = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "used", "which"}
    key_terms = gt_words - stopwords

    if not key_terms:
        key_terms = gt_words

    matched_terms = [w for w in key_terms if w in gen]
    overlap_ratio = len(matched_terms) / max(1, len(key_terms))

    # Scoring decision
    if overlap_ratio >= 0.40 or (len(key_terms) <= 3 and len(matched_terms) >= 1 and len(words) >= 4):
        return 2, f"Correctly followed: matched key terms ({', '.join(matched_terms[:4])})."
    elif overlap_ratio >= 0.15 or len(matched_terms) >= 1 or len(words) >= 5:
        return 1, f"Partially followed: partial keyword match ({', '.join(matched_terms[:2]) if matched_terms else 'none'})."
    else:
        return 0, "Failed: Low relevance to ground truth answer."


def evaluate_model(
    model: MiniGPTChat,
    test_data: List[Dict[str, Any]],
    tokenizer: ChatTokenizer,
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluates instruction-following capability across a test dataset.
    Returns composite score, category breakdown, and itemized results.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    gen_cfg = generation_config or GenerationConfig(max_new_tokens=48, temperature=0.3, do_sample=False)

    total_points = 0
    max_points = len(test_data) * 2
    category_stats: Dict[str, Dict[str, int]] = {}
    detailed_results: List[Dict[str, Any]] = []

    for item in test_data:
        messages = item["messages"]
        category = item.get("category", "general")

        if category not in category_stats:
            category_stats[category] = {"points": 0, "max_points": 0, "count": 0}

        # Find user prompt and assistant target
        user_prompt = ""
        ground_truth = ""
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
            elif msg["role"] == "assistant":
                ground_truth = msg["content"]

        # Prompt format
        prompt_msgs = [
            {"role": "system", "content": "You are a helpful, accurate, and concise AI assistant."},
            {"role": "user", "content": user_prompt}
        ]
        formatted = format_chat(prompt_msgs, add_generation_prompt=True)
        prompt_ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long, device=device)

        with torch.no_grad():
            gen_ids = model.generate(
                prompt_ids,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_k=gen_cfg.top_k,
                top_p=gen_cfg.top_p,
                stop_token_id=tokenizer.end_id,
                do_sample=gen_cfg.do_sample,
                repetition_penalty=getattr(gen_cfg, "repetition_penalty", 1.2)
            )

        full_output = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=False)
        assistant_reply = extract_assistant_response(full_output)

        score, rationale = score_response(assistant_reply, ground_truth, user_prompt)
        total_points += score

        category_stats[category]["points"] += score
        category_stats[category]["max_points"] += 2
        category_stats[category]["count"] += 1

        detailed_results.append({
            "instruction": user_prompt,
            "ground_truth": ground_truth,
            "generated": assistant_reply,
            "score": score,
            "rationale": rationale,
            "category": category
        })

    overall_score_pct = round((total_points / max(1, max_points)) * 100, 2)

    cat_breakdown = {}
    for cat, stats in category_stats.items():
        cat_pct = round((stats["points"] / max(1, stats["max_points"])) * 100, 2)
        cat_breakdown[cat] = {
            "score_pct": cat_pct,
            "points": stats["points"],
            "max_points": stats["max_points"],
            "count": stats["count"]
        }

    return {
        "overall_score_pct": overall_score_pct,
        "total_points": total_points,
        "max_points": max_points,
        "evaluated_prompts": len(test_data),
        "category_breakdown": cat_breakdown,
        "results": detailed_results
    }


def compare_base_vs_sft(
    base_model: MiniGPTChat,
    sft_model: MiniGPTChat,
    prompts: List[Dict[str, str]],
    tokenizer: ChatTokenizer,
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """
    Direct head-to-head comparison between pre-SFT base model and post-SFT model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device).eval()
    sft_model = sft_model.to(device).eval()

    gen_cfg = GenerationConfig(max_new_tokens=48, temperature=0.2, do_sample=False, repetition_penalty=1.2)
    comparisons: List[Dict[str, Any]] = []

    for item in prompts:
        prompt = item["instruction"]
        gt = item.get("ground_truth", "")
        category = item.get("category", "general")

        prompt_msgs = [
            {"role": "system", "content": "You are a helpful, accurate, and concise AI assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted = format_chat(prompt_msgs, add_generation_prompt=True)
        prompt_ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long, device=device)

        with torch.no_grad():
            out_base = base_model.generate(prompt_ids, max_new_tokens=gen_cfg.max_new_tokens, temperature=0.2, do_sample=False, repetition_penalty=1.2)
            out_sft = sft_model.generate(prompt_ids, max_new_tokens=gen_cfg.max_new_tokens, temperature=0.2, do_sample=False, repetition_penalty=1.2)

        resp_base = extract_assistant_response(tokenizer.decode(out_base[0].tolist()))
        resp_sft = extract_assistant_response(tokenizer.decode(out_sft[0].tolist()))

        score_base, _ = score_response(resp_base, gt, prompt)
        score_sft, _ = score_response(resp_sft, gt, prompt)

        comparisons.append({
            "prompt": prompt,
            "category": category,
            "base_response": resp_base,
            "base_score": score_base,
            "sft_response": resp_sft,
            "sft_score": score_sft,
            "improvement": "Yes" if score_sft > score_base else ("Equal" if score_sft == score_base else "Regressed")
        })

    return comparisons


def save_evaluation_results(results: Dict[str, Any], filepath: Path) -> None:
    """Saves structured evaluation results to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
