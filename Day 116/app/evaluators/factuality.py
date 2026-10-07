"""
Factuality, grounded evidence verification, and hallucination / abstention evaluators.
"""
import re
from typing import Dict, List, Optional, Any, Union
from app.evaluators.exact_match import normalize_text


ABSTENTION_PATTERNS = [
    r"do not have enough information",
    r"does not (provide|contain|mention|state)",
    r"not mentioned in the (provided )?context",
    r"cannot (answer|determine|be determined)",
    r"insufficient information",
    r"not provided in the (given )?text",
    r"i (do not|cannot) know",
    r"not found in the (provided )?document",
    r"no information provided"
]


def check_abstention(text: str) -> bool:
    """Detects whether model response appropriately abstains from answering."""
    norm = text.lower()
    for pattern in ABSTENTION_PATTERNS:
        if re.search(pattern, norm):
            return True
    return False


def evaluate_factuality_example(
    prediction: str,
    reference: str,
    context: Optional[str] = None,
    supported: Optional[bool] = True
) -> Dict[str, Any]:
    """
    Evaluates a single factuality or hallucination test item:
    - If supported == True: model is expected to provide the correct fact grounded in context.
    - If supported == False: model is expected to abstain; inventing an answer is marked as a hallucination.
    """
    pred_clean = normalize_text(prediction)
    ref_clean = normalize_text(reference)
    abstained = check_abstention(prediction)

    if supported is False:
        # Unsupported question: abstention is correct (score 1.0), answering is hallucination (score 0.0)
        is_hallucination = not abstained
        score = 1.0 if abstained else 0.0
        return {
            "is_correct": abstained,
            "is_hallucination": is_hallucination,
            "abstained": abstained,
            "score": score,
            "details": "Appropriately abstained" if abstained else "Hallucinated unsupported claim"
        }

    # Supported question: check if key reference answer is present in prediction
    if not ref_clean:
        return {"is_correct": False, "is_hallucination": False, "abstained": abstained, "score": 0.0, "details": "Empty reference"}

    ref_tokens = ref_clean.split()
    matched_tokens = sum(1 for tok in ref_tokens if tok in pred_clean)
    overlap_ratio = matched_tokens / max(1, len(ref_tokens))

    is_correct = (ref_clean in pred_clean) or (overlap_ratio >= 0.75)
    score = 1.0 if is_correct else (0.5 if overlap_ratio >= 0.5 else 0.0)

    return {
        "is_correct": is_correct,
        "is_hallucination": False,
        "abstained": abstained,
        "score": score,
        "overlap_ratio": round(overlap_ratio, 4),
        "details": "Accurate grounded answer" if is_correct else "Incorrect fact provided"
    }


def evaluate_factuality_dataset(
    predictions: List[str],
    references: List[str],
    contexts: Optional[List[Optional[str]]] = None,
    supported_flags: Optional[List[Optional[bool]]] = None
) -> Dict[str, float]:
    """
    Evaluates an entire factuality dataset, computing:
    - supported_accuracy: accuracy on questions where answer was in context
    - unsupported_abstention_rate: rate of abstaining on unanswerable questions
    - hallucination_rate: rate of generating hallucinated answers on unsupported questions
    - overall_factuality_score: composite factuality percentage
    """
    if not predictions:
        return {
            "supported_accuracy": 0.0,
            "unsupported_abstention_rate": 0.0,
            "hallucination_rate": 0.0,
            "overall_factuality_score": 0.0
        }

    contexts = contexts or [None] * len(predictions)
    supported_flags = supported_flags or [True] * len(predictions)

    sup_total, sup_correct = 0, 0
    unsup_total, unsup_abstained, unsup_hallucinated = 0, 0, 0

    for p, r, ctx, sup in zip(predictions, references, contexts, supported_flags):
        res = evaluate_factuality_example(p, r, context=ctx, supported=sup)
        if sup is False:
            unsup_total += 1
            if res["abstained"]:
                unsup_abstained += 1
            if res["is_hallucination"]:
                unsup_hallucinated += 1
        else:
            sup_total += 1
            if res["is_correct"]:
                sup_correct += 1

    sup_acc = (sup_correct / sup_total) if sup_total > 0 else 0.0
    unsup_abs = (unsup_abstained / unsup_total) if unsup_total > 0 else 0.0
    halluc_rate = (unsup_hallucinated / unsup_total) if unsup_total > 0 else 0.0

    # Composite score: average of supported accuracy and unsupported abstention
    if sup_total > 0 and unsup_total > 0:
        composite = 0.5 * sup_acc + 0.5 * unsup_abs
    elif sup_total > 0:
        composite = sup_acc
    else:
        composite = unsup_abs

    return {
        "supported_accuracy": round(sup_acc * 100.0, 2),
        "unsupported_abstention_rate": round(unsup_abs * 100.0, 2),
        "hallucination_rate": round(halluc_rate * 100.0, 2),
        "overall_factuality_score": round(composite * 100.0, 2)
    }
