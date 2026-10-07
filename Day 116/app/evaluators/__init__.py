"""
Evaluators module for Day 116: Exact match, BLEU, ROUGE, and Semantic metrics.
"""
from app.evaluators.exact_match import (
    exact_match,
    normalized_exact_match,
    token_accuracy,
    char_error_rate,
    normalize_text
)
from app.evaluators.bleu import (
    bleu_score,
    sentence_bleu,
    corpus_bleu
)
from app.evaluators.rouge import (
    rouge_n,
    rouge_l,
    compute_rouge_all,
    corpus_rouge
)
from app.evaluators.semantic import (
    cosine_similarity,
    semantic_similarity,
    bertscore_simulation
)
from app.evaluators.factuality import (
    evaluate_factuality_example,
    evaluate_factuality_dataset,
    check_abstention
)
from app.evaluators.safety import (
    evaluate_safety_example,
    evaluate_safety_dataset,
    check_safety_refusal,
    check_educational_pivot
)
from app.evaluators.coding import (
    extract_code,
    validate_syntax,
    execute_test_cases,
    evaluate_coding_example
)
from app.evaluators.instruction import (
    validate_json_output,
    count_bullet_points,
    evaluate_instruction_following
)
from app.evaluators.preference import (
    pairwise_win_rate,
    judge_response_quality,
    compare_two_responses,
    check_position_bias
)

__all__ = [
    "exact_match",
    "normalized_exact_match",
    "token_accuracy",
    "char_error_rate",
    "normalize_text",
    "bleu_score",
    "sentence_bleu",
    "corpus_bleu",
    "rouge_n",
    "rouge_l",
    "compute_rouge_all",
    "corpus_rouge",
    "cosine_similarity",
    "semantic_similarity",
    "bertscore_simulation",
    "evaluate_factuality_example",
    "evaluate_factuality_dataset",
    "check_abstention",
    "evaluate_safety_example",
    "evaluate_safety_dataset",
    "check_safety_refusal",
    "check_educational_pivot",
    "extract_code",
    "validate_syntax",
    "execute_test_cases",
    "evaluate_coding_example",
    "validate_json_output",
    "count_bullet_points",
    "evaluate_instruction_following",
    "pairwise_win_rate",
    "judge_response_quality",
    "compare_two_responses",
    "check_position_bias"
]
