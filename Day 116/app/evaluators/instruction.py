"""
Instruction-following constraint evaluators: structured JSON, bullet points, and formatting rules.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple


def validate_json_output(text: str, required_keys: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validates whether the output contains valid JSON and has any required keys.
    Extracts text between first { and last }, or first [ and last ].
    """
    clean = text.strip()
    # Try finding JSON substring
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
    if json_match:
        clean = json_match.group(1)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        return False, f"JSON parsing failed: {e}"

    if required_keys and isinstance(data, dict):
        missing = [k for k in required_keys if k not in data]
        if missing:
            return False, f"Missing required JSON keys: {missing}"

    return True, None


def count_bullet_points(text: str) -> int:
    """Counts bullet points starting with -, *, or numbered items (e.g., '1.', '2.')."""
    lines = text.strip().split("\n")
    count = 0
    for line in lines:
        s = line.strip()
        if re.match(r"^(\-|\*|\•|\d+\.)\s+", s):
            count += 1
    return count


def evaluate_instruction_following(prompt: str, prediction: str) -> Dict[str, Any]:
    """
    Analyzes whether the model strictly obeyed formatting and structural constraints
    specified in the user's prompt.
    Returns:
      {"score": 0-100, "details": [...], "violations": [...]}
    """
    prompt_lower = prompt.lower()
    pred_clean = prediction.strip()
    score = 100.0
    violations = []
    details = []

    # 1. JSON constraint
    if "json" in prompt_lower:
        is_json, err = validate_json_output(pred_clean)
        if not is_json:
            score -= 50.0
            violations.append(f"Failed JSON constraint: {err}")
        else:
            details.append("Valid JSON format detected.")

    # 2. Bullet count constraint
    bullet_match = re.search(r"(exactly|provide|list)\s+(two|three|four|five|\d+)\s+bullet", prompt_lower)
    word_to_num = {"two": 2, "three": 3, "four": 4, "five": 5}
    if bullet_match:
        target_str = bullet_match.group(2)
        target_count = word_to_num.get(target_str, int(target_str) if target_str.isdigit() else 3)
        actual_bullets = count_bullet_points(pred_clean)
        if actual_bullets != target_count:
            score -= 40.0
            violations.append(f"Expected {target_count} bullets, found {actual_bullets}.")
        else:
            details.append(f"Successfully provided exactly {target_count} bullets.")

    # 3. Exactly one word / four words constraint
    if "exactly one word" in prompt_lower or "one word" in prompt_lower:
        words = pred_clean.split()
        if len(words) != 1:
            score -= 40.0
            violations.append(f"Expected exactly 1 word, got {len(words)} words.")
        else:
            details.append("Single word constraint satisfied.")

    if "exactly four words" in prompt_lower:
        words = pred_clean.split()
        if len(words) != 4:
            score -= 40.0
            violations.append(f"Expected 4 words, got {len(words)} words.")
        else:
            details.append("Four words constraint satisfied.")

    # 4. Capital letters constraint
    if "capital letters" in prompt_lower or "all caps" in prompt_lower:
        alpha_chars = [c for c in pred_clean if c.isalpha()]
        if alpha_chars and not all(c.isupper() for c in alpha_chars):
            score -= 40.0
            violations.append("Output was not entirely in capital letters.")
        else:
            details.append("All-caps constraint satisfied.")

    # 5. Starting phrase constraint
    start_match = re.search(r"start (your response )?with ['\"]([^'\"]+)['\"]", prompt_lower)
    if start_match:
        req_start = start_match.group(2)
        if not pred_clean.lower().startswith(req_start.lower()):
            score -= 30.0
            violations.append(f"Did not start with required prefix '{req_start}'.")
        else:
            details.append(f"Prefix '{req_start}' matched.")

    # 6. Ending phrase constraint
    end_match = re.search(r"end (your response )?with (the phrase )?['\"]([^'\"]+)['\"]", prompt_lower)
    if end_match:
        req_end = end_match.group(3)
        if not pred_clean.lower().endswith(req_end.lower()):
            score -= 30.0
            violations.append(f"Did not end with required suffix '{req_end}'.")
        else:
            details.append(f"Suffix '{req_end}' matched.")

    final_score = max(0.0, min(100.0, score))
    return {
        "score": final_score,
        "is_compliant": final_score >= 80.0,
        "violations": violations,
        "details": details
    }
