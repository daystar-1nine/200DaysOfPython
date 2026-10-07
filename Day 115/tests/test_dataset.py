"""
Unit tests for preference dataset generation, validation, formatting, and file I/O.
"""
import pytest
from pathlib import Path

from app.data.formatter import (
    validate_preference_pair,
    format_prompt,
    format_preference_pair,
    extract_assistant_response,
    SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN
)
from app.data.preference_dataset import (
    expand_preference_dataset,
    split_preference_dataset,
    save_jsonl,
    load_jsonl
)


def test_valid_preference_pair(sample_preference_pair):
    assert validate_preference_pair(sample_preference_pair) is True


def test_missing_prompt_fails_validation():
    item = {"chosen": "good", "rejected": "bad"}
    assert validate_preference_pair(item) is False


def test_missing_chosen_fails_validation():
    item = {"prompt": "query", "rejected": "bad"}
    assert validate_preference_pair(item) is False


def test_missing_rejected_fails_validation():
    item = {"prompt": "query", "chosen": "good"}
    assert validate_preference_pair(item) is False


def test_non_dict_input_fails_validation():
    assert validate_preference_pair(["prompt", "chosen", "rejected"]) is False
    assert validate_preference_pair("invalid string") is False
    assert validate_preference_pair(None) is False


def test_empty_string_prompt_fails_validation():
    item = {"prompt": "", "chosen": "good", "rejected": "bad"}
    assert validate_preference_pair(item) is False


def test_empty_string_chosen_fails_validation():
    item = {"prompt": "query", "chosen": "", "rejected": "bad"}
    assert validate_preference_pair(item) is False


def test_empty_string_rejected_fails_validation():
    item = {"prompt": "query", "chosen": "good", "rejected": ""}
    assert validate_preference_pair(item) is False


def test_whitespace_only_fails_validation():
    item = {"prompt": "   ", "chosen": "good", "rejected": "bad"}
    assert validate_preference_pair(item) is False


def test_identical_chosen_and_rejected_fails_validation():
    item = {"prompt": "query", "chosen": "same answer", "rejected": "same answer"}
    assert validate_preference_pair(item) is False


def test_format_prompt_tags_structure():
    prompt = "Hello"
    fmt = format_prompt(prompt, system_prompt="You are helpful.", add_generation_prompt=True)
    assert fmt.startswith(SYSTEM_TOKEN)
    assert USER_TOKEN in fmt
    assert fmt.endswith(ASSISTANT_TOKEN)
    assert f"Hello{END_TOKEN}" in fmt


def test_format_preference_pair_outputs():
    prompt = "What is 2+2?"
    chosen = "4"
    rejected = "5"
    p_fmt, c_seq, r_seq = format_preference_pair(prompt, chosen, rejected)

    assert p_fmt.endswith(ASSISTANT_TOKEN)
    assert c_seq.startswith(p_fmt)
    assert c_seq.endswith(f"4{END_TOKEN}")
    assert r_seq.startswith(p_fmt)
    assert r_seq.endswith(f"5{END_TOKEN}")


def test_extract_assistant_response_clean():
    raw = f"{SYSTEM_TOKEN}Sys{END_TOKEN}{USER_TOKEN}Hi{END_TOKEN}{ASSISTANT_TOKEN}Hello world!{END_TOKEN}"
    res = extract_assistant_response(raw)
    assert res == "Hello world!"


def test_extract_assistant_response_without_end_token():
    raw = f"{SYSTEM_TOKEN}Sys{END_TOKEN}{USER_TOKEN}Hi{END_TOKEN}{ASSISTANT_TOKEN}Unfinished reply"
    res = extract_assistant_response(raw)
    assert res == "Unfinished reply"


def test_extract_assistant_response_no_assistant_tag():
    raw = "Direct plain text response"
    res = extract_assistant_response(raw)
    assert res == "Direct plain text response"


def test_expand_preference_dataset_volume_and_categories():
    pairs = expand_preference_dataset(multiplier=12, seed=42)
    assert len(pairs) >= 500
    categories = {p["category"] for p in pairs}
    expected_categories = {"correctness", "conciseness", "helpfulness", "safety", "instruction_following"}
    assert categories == expected_categories


def test_split_preference_dataset_proportions():
    pairs = expand_preference_dataset(multiplier=12, seed=42)
    total = len(pairs)
    train_set, val_set, test_set = split_preference_dataset(pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert len(train_set) + len(val_set) + len(test_set) == total
    assert round(len(train_set) / total, 1) == 0.8
    assert len(val_set) > 0
    assert len(test_set) > 0


def test_split_preference_dataset_determinism():
    pairs = expand_preference_dataset(multiplier=2, seed=42)
    t1, v1, te1 = split_preference_dataset(pairs, seed=10)
    t2, v2, te2 = split_preference_dataset(pairs, seed=10)
    assert [x["prompt"] for x in t1] == [x["prompt"] for x in t2]


def test_save_and_load_jsonl_roundtrip(tmp_path):
    filepath = tmp_path / "test_prefs.jsonl"
    data = [
        {"prompt": "P1", "chosen": "C1", "rejected": "R1"},
        {"prompt": "P2", "chosen": "C2", "rejected": "R2"}
    ]
    save_jsonl(data, filepath)
    assert filepath.exists()

    loaded = load_jsonl(filepath)
    assert len(loaded) == 2
    assert loaded[0]["prompt"] == "P1"
    assert loaded[1]["chosen"] == "C2"


def test_load_nonexistent_jsonl_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nonexistent.jsonl")
