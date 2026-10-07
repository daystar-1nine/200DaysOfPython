"""
Evaluation utilities for calculating pairwise preference accuracy and reward margins
for both Reward Models and Generative Policy models (via response log probabilities).
"""
from typing import List, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from app.models.reward_model import RewardModel
from app.models.dpo_model import MiniGPTChat
from app.data.tokenizer import ChatTokenizer
from app.data.collator import PreferenceCollator


def evaluate_reward_model_preferences(
    reward_model: RewardModel,
    dataset: List[Dict[str, Any]],
    tokenizer: ChatTokenizer,
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluates Reward Model ranking accuracy:
      Accuracy = (r_chosen > r_rejected).mean()
    Also breaks down accuracy per category.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = reward_model.to(device)
    reward_model.eval()
    cfg = getattr(reward_model, "model_config", getattr(reward_model, "config", None))
    ctx_len = getattr(cfg, "context_length", getattr(getattr(cfg, "backbone_config", None), "context_length", 128))
    collator = PreferenceCollator(tokenizer, max_length=ctx_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    total_correct = 0
    total_count = 0
    total_margin = 0.0
    category_stats: Dict[str, Dict[str, int]] = {}

    idx = 0
    with torch.no_grad():
        for batch in loader:
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)

            rc, rr = reward_model.forward_pairwise(c_ids, c_mask, r_ids, r_mask)
            is_correct = (rc > rr).cpu().tolist()
            margins = (rc - rr).cpu().tolist()

            for c, m in zip(is_correct, margins):
                cat = dataset[idx].get("category", "general")
                category_stats.setdefault(cat, {"correct": 0, "total": 0})
                if c:
                    total_correct += 1
                    category_stats[cat]["correct"] += 1
                category_stats[cat]["total"] += 1
                total_margin += m
                total_count += 1
                idx += 1

    overall_acc = round((total_correct / max(1, total_count)) * 100, 2)
    mean_margin = round(total_margin / max(1, total_count), 4)

    cat_breakdown = {}
    for cat, stats in category_stats.items():
        cat_acc = round((stats["correct"] / max(1, stats["total"])) * 100, 2)
        cat_breakdown[cat] = {
            "accuracy": cat_acc,
            "correct": stats["correct"],
            "total": stats["total"]
        }

    return {
        "accuracy": overall_acc,
        "mean_margin": mean_margin,
        "total_evaluated": total_count,
        "category_breakdown": cat_breakdown
    }


def evaluate_policy_preferences(
    policy_model: MiniGPTChat,
    dataset: List[Dict[str, Any]],
    tokenizer: ChatTokenizer,
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluates generative policy preference accuracy based on response log-probabilities:
      Accuracy = (log P_theta(y_c | x) > log P_theta(y_r | x)).mean()
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model = policy_model.to(device)
    policy_model.eval()

    ctx_len = getattr(getattr(policy_model, "config", None), "context_length", 128)
    collator = PreferenceCollator(tokenizer, max_length=ctx_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    total_correct = 0
    total_count = 0
    total_log_margin = 0.0
    category_stats: Dict[str, Dict[str, int]] = {}

    idx = 0
    with torch.no_grad():
        for batch in loader:
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            c_labels = batch["chosen_labels"].to(device)

            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            r_labels = batch["rejected_labels"].to(device)

            c_logps = policy_model.get_response_log_probs(c_ids, attention_mask=c_mask, labels=c_labels)
            r_logps = policy_model.get_response_log_probs(r_ids, attention_mask=r_mask, labels=r_labels)

            is_correct = (c_logps > r_logps).cpu().tolist()
            log_margins = (c_logps - r_logps).cpu().tolist()

            for c, lm in zip(is_correct, log_margins):
                cat = dataset[idx].get("category", "general")
                category_stats.setdefault(cat, {"correct": 0, "total": 0})
                if c:
                    total_correct += 1
                    category_stats[cat]["correct"] += 1
                category_stats[cat]["total"] += 1
                total_log_margin += lm
                total_count += 1
                idx += 1

    overall_acc = round((total_correct / max(1, total_count)) * 100, 2)
    mean_margin = round(total_log_margin / max(1, total_count), 4)

    cat_breakdown = {}
    for cat, stats in category_stats.items():
        cat_acc = round((stats["correct"] / max(1, stats["total"])) * 100, 2)
        cat_breakdown[cat] = {
            "accuracy": cat_acc,
            "correct": stats["correct"],
            "total": stats["total"]
        }

    return {
        "accuracy": overall_acc,
        "mean_log_margin": mean_margin,
        "total_evaluated": total_count,
        "category_breakdown": cat_breakdown
    }
