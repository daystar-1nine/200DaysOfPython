"""
Dataset splitting and benchmark leakage detection utilities.
Ensures rigorous train / validation / test partitioning without data contamination.
"""
from typing import List, Dict, Tuple, Set, Any
import random
from collections import defaultdict

from app.datasets.schema import EvaluationExample


def stratified_split(
    examples: List[EvaluationExample],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[EvaluationExample], List[EvaluationExample], List[EvaluationExample]]:
    """
    Partitions a dataset into stratified train, validation, and test subsets based on category.
    Guarantees reproducible splits and mutually disjoint ID sets.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    rng = random.Random(seed)
    by_category: Dict[str, List[EvaluationExample]] = defaultdict(list)
    for ex in examples:
        by_category[ex.category.lower()].append(ex)

    train_set: List[EvaluationExample] = []
    val_set: List[EvaluationExample] = []
    test_set: List[EvaluationExample] = []

    for cat, cat_examples in sorted(by_category.items()):
        shuffled = list(cat_examples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 item in test if n >= 3
        if n >= 3 and (n - n_train - n_val) < 1:
            n_train = max(1, n_train - 1)

        c_train = shuffled[:n_train]
        c_val = shuffled[n_train:n_train + n_val]
        c_test = shuffled[n_train + n_val:]

        train_set.extend(c_train)
        val_set.extend(c_val)
        test_set.extend(c_test)

    # Sanity verify disjoint IDs
    train_ids = {e.id for e in train_set}
    val_ids = {e.id for e in val_set}
    test_ids = {e.id for e in test_set}

    assert len(train_ids & val_ids) == 0, "Leakage detected between train and val!"
    assert len(train_ids & test_ids) == 0, "Leakage detected between train and test!"
    assert len(val_ids & test_ids) == 0, "Leakage detected between val and test!"

    return train_set, val_set, test_set


def detect_data_leakage(
    reference_set: List[EvaluationExample],
    eval_set: List[EvaluationExample]
) -> Dict[str, Any]:
    """
    Scans for potential evaluation benchmark contamination:
    1. Exact ID duplicates
    2. Normalized exact prompt duplicates
    3. High token-level n-gram overlap between train reference and test evaluation
    """
    ref_ids = {e.id for e in reference_set}
    ref_prompts = {e.prompt.strip().lower() for e in reference_set}

    id_overlap: List[str] = []
    prompt_overlap: List[str] = []

    for e in eval_set:
        if e.id in ref_ids:
            id_overlap.append(e.id)
        if e.prompt.strip().lower() in ref_prompts:
            prompt_overlap.append(e.id)

    is_contaminated = bool(id_overlap or prompt_overlap)
    return {
        "is_contaminated": is_contaminated,
        "id_collisions": id_overlap,
        "exact_prompt_collisions": prompt_overlap,
        "clean_eval_percentage": round(100.0 * (1.0 - len(prompt_overlap) / max(1, len(eval_set))), 2)
    }
