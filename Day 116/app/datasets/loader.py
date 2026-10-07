"""
Data loader and persistence utilities for benchmark JSONL files.
"""
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import json

from app.datasets.schema import EvaluationExample


def load_jsonl_dataset(filepath: Path, validate: bool = True) -> List[EvaluationExample]:
    """
    Loads evaluation examples from a JSONL file.
    Validates each record against the schema if validate=True.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {filepath}")

    examples: List[EvaluationExample] = []
    seen_ids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {filepath} line {line_num}: {e}")

            example = EvaluationExample.from_dict(data)
            if validate:
                example.validate()

            if example.id in seen_ids:
                raise ValueError(f"Duplicate example ID '{example.id}' found at line {line_num} in {filepath}")
            seen_ids.add(example.id)
            examples.append(example)

    return examples


load_evaluation_dataset = load_jsonl_dataset


def save_jsonl_dataset(examples: List[EvaluationExample], filepath: Path) -> None:
    """
    Saves a list of EvaluationExample items to a JSONL file.
    Creates parent directories if missing.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            json_str = json.dumps(ex.to_dict(), ensure_ascii=False)
            f.write(json_str + "\n")


def filter_by_category(examples: List[EvaluationExample], category: str) -> List[EvaluationExample]:
    """Returns subset of examples matching a target category (case-insensitive)."""
    target = category.strip().lower()
    return [ex for ex in examples if ex.category.lower() == target]


def validate_dataset(examples: List[EvaluationExample]) -> Dict[str, Any]:
    """
    Audits an in-memory dataset collection:
    - Counts per category
    - Verified unique IDs
    - Returns summary dictionary
    """
    if not examples:
        raise ValueError("Cannot validate empty dataset.")

    category_counts: Dict[str, int] = {}
    seen_ids = set()

    for idx, ex in enumerate(examples):
        ex.validate()
        if ex.id in seen_ids:
            raise ValueError(f"Duplicate ID '{ex.id}' detected at index {idx}.")
        seen_ids.add(ex.id)
        cat = ex.category.lower()
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total_examples": len(examples),
        "unique_ids": len(seen_ids),
        "category_counts": category_counts
    }


def batch_examples(
    examples: List[EvaluationExample],
    batch_size: int = 16
) -> Generator[List[EvaluationExample], None, None]:
    """Yields consecutive batches of evaluation examples."""
    for i in range(0, len(examples), batch_size):
        yield examples[i:i + batch_size]
