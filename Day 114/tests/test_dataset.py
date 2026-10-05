"""
Unit tests for Day 114 Dataset, Formatter, and Validation modules.
"""
import sys
import tempfile
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.data.formatter import (
    validate_conversation,
    format_chat,
    extract_assistant_response,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_TOKEN
)
from app.data.dataset import (
    load_jsonl,
    save_jsonl,
    split_dataset,
    generate_raw_instruction_pairs,
    build_instruction_dataset,
    CATEGORIES
)


class TestConversationValidation:
    def test_valid_single_turn_conversation(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        assert validate_conversation(msgs) is True

    def test_valid_conversation_without_system(self):
        msgs = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        assert validate_conversation(msgs) is True

    def test_invalid_empty_list(self):
        assert validate_conversation([]) is False

    def test_invalid_non_list_input(self):
        assert validate_conversation("Not a list") is False
        assert validate_conversation({"role": "user", "content": "Hi"}) is False

    def test_invalid_item_not_dict(self):
        assert validate_conversation(["item1", "item2"]) is False

    def test_invalid_missing_keys(self):
        assert validate_conversation([{"role": "user"}]) is False
        assert validate_conversation([{"content": "Hello"}]) is False

    def test_invalid_empty_or_whitespace_content(self):
        assert validate_conversation([{"role": "user", "content": ""}]) is False
        assert validate_conversation([{"role": "user", "content": "   \n\t"}]) is False

    def test_invalid_role_name(self):
        assert validate_conversation([{"role": "moderator", "content": "Warning"}]) is False

    def test_invalid_system_not_first(self):
        msgs = [
            {"role": "user", "content": "Hello!"},
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "Hi there!"}
        ]
        assert validate_conversation(msgs) is False

    def test_invalid_missing_user_message(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "Hi there!"}
        ]
        assert validate_conversation(msgs) is False


class TestChatFormatter:
    def test_format_chat_tags_presence(self, sample_conversations):
        formatted = format_chat(sample_conversations[0]["messages"])
        assert SYSTEM_TOKEN in formatted
        assert USER_TOKEN in formatted
        assert ASSISTANT_TOKEN in formatted
        assert END_TOKEN in formatted

    def test_format_chat_generation_prompt(self):
        msgs = [{"role": "user", "content": "Explain recursion."}]
        formatted = format_chat(msgs, add_generation_prompt=True)
        assert formatted.endswith(ASSISTANT_TOKEN)

    def test_format_chat_unsupported_role_raises(self):
        msgs = [{"role": "bot", "content": "Hello"}]
        with pytest.raises(ValueError, match="Unsupported message role"):
            format_chat(msgs)

    def test_extract_assistant_response_clean(self):
        raw = f"{SYSTEM_TOKEN}System{END_TOKEN}{USER_TOKEN}User{END_TOKEN}{ASSISTANT_TOKEN}Expected output.{END_TOKEN}"
        extracted = extract_assistant_response(raw)
        assert extracted == "Expected output."

    def test_extract_assistant_response_without_end_token(self):
        raw = f"{ASSISTANT_TOKEN}Partial response"
        extracted = extract_assistant_response(raw)
        assert extracted == "Partial response"

    def test_extract_assistant_response_no_assistant_tag(self):
        raw = "Direct text output"
        assert extract_assistant_response(raw) == "Direct text output"


class TestJSONLLoadAndSave:
    def test_save_and_load_roundtrip(self, sample_conversations):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test.jsonl"
            save_jsonl(sample_conversations, fpath)
            assert fpath.exists()

            loaded = load_jsonl(fpath)
            assert len(loaded) == len(sample_conversations)
            assert loaded[0]["messages"][1]["content"] == sample_conversations[0]["messages"][1]["content"]

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl(Path("nonexistent_path_42.jsonl"))

    def test_load_malformed_json_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "malformed.jsonl"
            fpath.write_text("{bad json line\n", encoding="utf-8")
            with pytest.raises(ValueError, match="Malformed JSON"):
                load_jsonl(fpath)

    def test_load_missing_messages_key_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "no_msg.jsonl"
            fpath.write_text('{"prompt": "hi", "completion": "hello"}\n', encoding="utf-8")
            with pytest.raises(ValueError, match="Missing 'messages' key"):
                load_jsonl(fpath)


class TestDatasetSplittingAndGeneration:
    def test_split_dataset_empty(self):
        train, val, test = split_dataset([])
        assert train == []
        assert val == []
        assert test == []

    def test_split_dataset_proportions(self):
        data = [{"item": i} for i in range(100)]
        train, val, test = split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_dataset_determinism(self):
        data = [{"item": i} for i in range(50)]
        t1, v1, s1 = split_dataset(data, seed=123)
        t2, v2, s2 = split_dataset(data, seed=123)
        assert t1 == t2
        assert v1 == v2
        assert s1 == s2

    def test_generate_raw_instruction_pairs_volume(self):
        pairs = generate_raw_instruction_pairs()
        assert len(pairs) >= 500
        for p in pairs:
            assert "instruction" in p and len(p["instruction"].strip()) > 0
            assert "response" in p and len(p["response"].strip()) > 0
            assert "category" in p and p["category"] in CATEGORIES

    def test_all_categories_represented(self):
        pairs = generate_raw_instruction_pairs()
        found_cats = {p["category"] for p in pairs}
        for cat in CATEGORIES:
            assert cat in found_cats

    def test_build_instruction_dataset_format(self):
        dataset = build_instruction_dataset(system_prompt="Custom System Prompt")
        assert len(dataset) >= 500
        for item in dataset[:10]:
            assert "messages" in item
            assert len(item["messages"]) == 3
            assert item["messages"][0]["content"] == "Custom System Prompt"
            assert validate_conversation(item["messages"]) is True
