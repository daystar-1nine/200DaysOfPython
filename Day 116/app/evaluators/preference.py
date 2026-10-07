"""
Pairwise preference evaluation, win-rate computation, and LLM-as-a-judge bias diagnostics.
"""
from typing import Dict, List, Any, Optional, Tuple
from app.evaluators.exact_match import normalize_text
from app.evaluators.semantic import semantic_similarity


def pairwise_win_rate(wins_a: int, wins_b: int, ties: int) -> Dict[str, float]:
    """
    Computes pairwise win rates with half-point tie allocation:
      WinRate(A) = (wins_a + 0.5 * ties) / total
    """
    total = wins_a + wins_b + ties
    if total == 0:
        return {"win_rate_a": 0.0, "win_rate_b": 0.0, "tie_rate": 0.0}

    wr_a = (wins_a + 0.5 * ties) / total
    wr_b = (wins_b + 0.5 * ties) / total
    tr = ties / total

    return {
        "win_rate_a": round(wr_a * 100.0, 2),
        "win_rate_b": round(wr_b * 100.0, 2),
        "tie_rate": round(tr * 100.0, 2),
        "total_comparisons": total
    }


def judge_response_quality(
    prompt: str,
    response: str,
    reference: Optional[str] = None
) -> float:
    """
    Scores response quality on a continuous scale [0.0, 100.0] based on:
    - Non-emptiness and minimal repetition
    - Semantic similarity to reference (if available)
    - Conciseness (penalizing repetitive bloating)
    """
    resp_clean = response.strip()
    if not resp_clean or len(resp_clean) < 3:
        return 0.0

    words = resp_clean.split()
    # Repetition loop penalty
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.35:
            return 10.0

    score = 50.0  # Baseline for coherent text

    if reference:
        sim = semantic_similarity(resp_clean, reference)
        score += sim * 40.0

    # Conciseness bonus / penalty
    if 3 <= len(words) <= 35:
        score += 10.0
    elif len(words) > 70:
        score -= 15.0

    return max(0.0, min(100.0, score))


def compare_two_responses(
    prompt: str,
    resp_a: str,
    resp_b: str,
    reference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares Response A vs Response B on a prompt.
    Returns:
      winner: 'A', 'B', or 'tie'
    """
    score_a = judge_response_quality(prompt, resp_a, reference)
    score_b = judge_response_quality(prompt, resp_b, reference)

    # Threshold for tie
    if abs(score_a - score_b) < 3.0:
        winner = "tie"
    elif score_a > score_b:
        winner = "A"
    else:
        winner = "B"

    return {
        "winner": winner,
        "score_a": round(score_a, 2),
        "score_b": round(score_b, 2)
    }


def check_position_bias(
    prompt: str,
    resp_a: str,
    resp_b: str,
    reference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates position bias by evaluating (A, B) and reversed (B, A).
    If order flips the preference, position bias is detected.
    """
    eval_forward = compare_two_responses(prompt, resp_a, resp_b, reference)
    eval_reverse = compare_two_responses(prompt, resp_b, resp_a, reference)

    forward_winner = eval_forward["winner"]
    reverse_winner = eval_reverse["winner"]

    # In reverse evaluation, 'A' corresponds to resp_b, 'B' corresponds to resp_a
    consistent = (
        (forward_winner == "A" and reverse_winner == "B") or
        (forward_winner == "B" and reverse_winner == "A") or
        (forward_winner == "tie" and reverse_winner == "tie")
    )

    return {
        "consistent": consistent,
        "has_position_bias": not consistent,
        "forward_winner": forward_winner,
        "reverse_winner": reverse_winner
    }
