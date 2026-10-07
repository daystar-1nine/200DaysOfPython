"""
Unit tests for benchmark dataset schema, loader, filtering, and splits.
"""
from pathlib import Path
import pytest
import json

from app.datasets.schema import EvaluationExample, VALID_CATEGORIES, VALID_TYPES
from app.datasets.loader import (
    load_jsonl_dataset,
    save_jsonl_dataset,
    filter_by_category,
    validate_dataset,
    batch_examples
)
from app.datasets.splits import stratified_split, detect_data_leakage


def test_valid_example_creation():
    ex = EvaluationExample(
        id="ex_001",
        category="python",
        expected_type="text",
        prompt="Write a list comprehension.",
        reference="[x for x in range(10)]"
    )
    assert ex.id == "ex_001"
    assert ex.category == "python"
    assert ex.expected_type == "text"
    assert ex.reference == "[x for x in range(10)]"
    assert ex.get_references() == ["[x for x in range(10)]"]


def test_example_missing_id_raises():
    ex = EvaluationExample(id="", category="python", expected_type="text", prompt="Test", reference="Ref")
    with pytest.raises(ValueError, match="Invalid or empty example ID"):
        ex.validate()


def test_example_missing_prompt_raises():
    ex = EvaluationExample(id="ex_1", category="python", expected_type="text", prompt="   ", reference="Ref")
    with pytest.raises(ValueError, match="Invalid or empty prompt"):
        ex.validate()


def test_example_missing_reference_raises():
    ex = EvaluationExample(id="ex_1", category="python", expected_type="text", prompt="Test", reference="", references=[])
    with pytest.raises(ValueError, match="has no valid reference answers"):
        ex.validate()


def test_example_invalid_category_raises():
    ex = EvaluationExample(id="ex_1", category="astrophysics_invalid", expected_type="text", prompt="Test", reference="Ref")
    with pytest.raises(ValueError, match="not in recognized categories"):
        ex.validate()


def test_example_invalid_type_raises():
    ex = EvaluationExample(id="ex_1", category="python", expected_type="teleportation_invalid", prompt="Test", reference="Ref")
    with pytest.raises(ValueError, match="not in recognized types"):
        ex.validate()


def test_example_roundtrip_dict():
    ex = EvaluationExample(
        id="ex_rt",
        category="machine_learning",
        expected_type="text",
        prompt="Define overfitting.",
        reference="A model memorizes noise instead of general patterns.",
        references=["A model memorizes noise instead of general patterns.", "High train accuracy, low val accuracy."],
        context="Overfitting happens when parameters exceed sample capacity.",
        supported=True,
        metadata={"tags": ["ml", "core"]}
    )
    d = ex.to_dict()
    ex2 = EvaluationExample.from_dict(d)
    assert ex2.id == ex.id
    assert ex2.category == ex.category
    assert ex2.metadata == {"tags": ["ml", "core"]}
    assert ex2.supported is True


def test_example_get_references_combines_properly():
    ex = EvaluationExample(
        id="ex_ref",
        category="reasoning",
        expected_type="text",
        prompt="Solve 2+2.",
        reference="4",
        references=["Four", "4", "4.0"]
    )
    refs = ex.get_references()
    assert "4" in refs
    assert "Four" in refs
    assert "4.0" in refs
    assert len(refs) == 3


def test_load_jsonl_dataset_success(temp_jsonl_file):
    loaded = load_jsonl_dataset(temp_jsonl_file)
    assert len(loaded) == 5
    assert loaded[0].id == "ex_python_01"


def test_load_jsonl_dataset_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_jsonl_dataset(Path("non_existent_dir/data.jsonl"))


def test_load_jsonl_dataset_duplicate_id_raises(tmp_path):
    dup_file = tmp_path / "dup.jsonl"
    with open(dup_file, "w") as f:
        f.write(json.dumps({"id": "dup1", "category": "python", "expected_type": "text", "prompt": "p1", "reference": "r1"}) + "\n")
        f.write(json.dumps({"id": "dup1", "category": "python", "expected_type": "text", "prompt": "p2", "reference": "r2"}) + "\n")
    with pytest.raises(ValueError, match="Duplicate example ID"):
        load_jsonl_dataset(dup_file)


def test_load_jsonl_dataset_corrupted_json_raises(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    with open(bad_file, "w") as f:
        f.write("{invalid json syntax\n")
    with pytest.raises(ValueError, match="JSON decode error"):
        load_jsonl_dataset(bad_file)


def test_save_jsonl_dataset_roundtrip(sample_examples, tmp_path):
    out_file = tmp_path / "saved.jsonl"
    save_jsonl_dataset(sample_examples, out_file)
    reloaded = load_jsonl_dataset(out_file)
    assert len(reloaded) == len(sample_examples)
    assert [ex.id for ex in reloaded] == [ex.id for ex in sample_examples]


def test_filter_by_category(sample_examples):
    fact_items = filter_by_category(sample_examples, "factuality")
    assert len(fact_items) == 2
    assert all(item.category == "factuality" for item in fact_items)

    case_items = filter_by_category(sample_examples, "PYTHON")
    assert len(case_items) == 1
    assert case_items[0].id == "ex_python_01"


def test_validate_dataset_summary(sample_examples):
    summary = validate_dataset(sample_examples)
    assert summary["total_examples"] == 5
    assert summary["unique_ids"] == 5
    assert summary["category_counts"]["factuality"] == 2
    assert summary["category_counts"]["coding"] == 1


def test_batch_examples_yields_correct_sizes(sample_examples):
    batches = list(batch_examples(sample_examples, batch_size=2))
    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1


def test_stratified_split_proportions(sample_examples):
    # Construct 20 valid items
    pool = []
    for i in range(20):
        src = sample_examples[i % len(sample_examples)]
        pool.append(EvaluationExample(
            id=f"item_{i}",
            category=src.category,
            expected_type=src.expected_type,
            prompt=f"{src.prompt} #{i}",
            reference=src.reference,
            supported=src.supported
        ))
    train_set, val_set, test_set = stratified_split(pool, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    assert len(train_set) + len(val_set) + len(test_set) == 20
    assert len(train_set) >= 10
    assert len(val_set) >= 2
    assert len(test_set) >= 2


def test_detect_data_leakage_flags_duplicate(sample_examples):
    train_set = [sample_examples[0], sample_examples[1]]
    test_set = [
        sample_examples[2],
        EvaluationExample(
            id="leak_01",
            category="python",
            expected_type="text",
            prompt="Explain Python list comprehension in one sentence.",
            reference="List comprehension syntax explanation."
        )
    ]
    leakage = detect_data_leakage(train_set, test_set)
    assert leakage["is_contaminated"] is True
    assert len(leakage["exact_prompt_collisions"]) == 1
