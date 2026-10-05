"""
Unit tests for SFTDataCollator and loss masking mechanics.
Verifies assistant-only loss masking, padding, attention masks, truncation,
and batch collation.
"""
import pytest
import torch

from app.data.tokenizer import ChatTokenizer
from app.data.collator import SFTDataCollator
from app.data.formatter import SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN


@pytest.fixture
def tokenizer():
    vocab = {
        "<|pad|>": 0,
        "<|end|>": 1,
        "<|system|>": 2,
        "<|user|>": 3,
        "<|assistant|>": 4,
        "a": 5, "b": 6, "c": 7, "d": 8, " ": 9, "?": 10, "!": 11, ".": 12
    }
    tok = ChatTokenizer()
    tok.vocab = vocab
    tok.inverse_vocab = {v: k for k, v in vocab.items()}
    return tok


@pytest.fixture
def sample_dialogue():
    return {
        "messages": [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "b?"},
            {"role": "assistant", "content": "c."}
        ]
    }


def test_collator_initialization(tokenizer):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=64, mask_prompt_loss=True)
    assert collator.tokenizer == tokenizer
    assert collator.max_length == 64
    assert collator.mask_prompt_loss is True


def test_collator_output_keys(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue])
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch


def test_collator_tensor_types(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue])
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_collator_tensor_shapes(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue, sample_dialogue])
    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].ndim == 2
    assert batch["labels"].ndim == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].size(0) == 2
    assert batch["input_ids"].size(1) == 32


def test_loss_masking_assistant_tokens_unmasked(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    # Unmasked tokens should have label == input_id (not -100)
    unmasked_indices = (labels != -100).nonzero(as_tuple=True)[0]
    assert len(unmasked_indices) > 0

    for idx in unmasked_indices:
        assert labels[idx].item() == input_ids[idx].item()


def test_loss_masking_prompt_tokens_masked(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # Find where assistant token is
    assistant_pos = (input_ids == tokenizer.assistant_id).nonzero(as_tuple=True)[0]
    assert len(assistant_pos) >= 1
    first_assistant_pos = assistant_pos[0].item()

    # All tokens up to and including the assistant token should be masked to -100
    for i in range(first_assistant_pos + 1):
        assert labels[i].item() == -100, f"Token at pos {i} ({input_ids[i].item()}) should be masked -100"


def test_loss_masking_assistant_end_token_unmasked(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # Assistant reply ends with END_TOKEN; the model must learn to predict END_TOKEN
    end_positions = (input_ids == tokenizer.end_id).nonzero(as_tuple=True)[0]
    assert len(end_positions) >= 1
    last_end = end_positions[-1].item()
    assert labels[last_end].item() == tokenizer.end_id


def test_system_and_user_end_tokens_masked(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    assistant_pos = (input_ids == tokenizer.assistant_id).nonzero(as_tuple=True)[0][0].item()
    end_positions = (input_ids == tokenizer.end_id).nonzero(as_tuple=True)[0]

    # End tokens before assistant prompt must be -100
    for pos in end_positions:
        if pos.item() < assistant_pos:
            assert labels[pos].item() == -100


def test_padding_and_attention_mask_alignment(tokenizer):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    item_short = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"}
        ]
    }
    item_long = {
        "messages": [
            {"role": "system", "content": "a b c"},
            {"role": "user", "content": "d a b?"},
            {"role": "assistant", "content": "c d a b."}
        ]
    }
    batch = collator([item_short, item_long])
    pad_id = tokenizer.pad_id

    short_input = batch["input_ids"][0]
    short_mask = batch["attention_mask"][0]
    short_labels = batch["labels"][0]

    pad_indices = (short_input == pad_id).nonzero(as_tuple=True)[0]
    assert len(pad_indices) > 0

    for idx in pad_indices:
        assert short_mask[idx].item() == 0, "Pad token must have attention mask 0"
        assert short_labels[idx].item() == -100, "Pad token must have label -100"

    valid_indices = (short_input != pad_id).nonzero(as_tuple=True)[0]
    for idx in valid_indices:
        assert short_mask[idx].item() == 1, "Non-pad token must have attention mask 1"


def test_truncation_to_max_seq_length(tokenizer):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=10)
    item = {
        "messages": [
            {"role": "system", "content": "a b c d"},
            {"role": "user", "content": "a b c d"},
            {"role": "assistant", "content": "a b c d"}
        ]
    }
    batch = collator([item])
    assert batch["input_ids"].size(1) == 10
    assert batch["attention_mask"].size(1) == 10
    assert batch["labels"].size(1) == 10


def test_no_prompt_masking_mode(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32, mask_prompt_loss=False)
    batch = collator([sample_dialogue])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    # Without prompt loss masking, all non-pad tokens should be unmasked
    for i in range(len(input_ids)):
        if input_ids[i].item() != tokenizer.pad_id:
            assert labels[i].item() == input_ids[i].item()


def test_multi_turn_conversation_masking(tokenizer):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    multi_turn = {
        "messages": [
            {"role": "user", "content": "a?"},
            {"role": "assistant", "content": "b."},
            {"role": "user", "content": "c?"},
            {"role": "assistant", "content": "d."}
        ]
    }
    batch = collator([multi_turn])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    user_indices = (input_ids == tokenizer.user_id).nonzero(as_tuple=True)[0]
    assistant_indices = (input_ids == tokenizer.assistant_id).nonzero(as_tuple=True)[0]
    assert len(user_indices) == 2
    assert len(assistant_indices) == 2

    # User markers themselves must be -100
    for uid in user_indices:
        assert labels[uid].item() == -100
    for aid in assistant_indices:
        assert labels[aid].item() == -100


def test_single_item_batch(tokenizer, sample_dialogue):
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=32)
    batch = collator([sample_dialogue])
    assert batch["input_ids"].size(0) == 1
    assert batch["attention_mask"].size(0) == 1
    assert batch["labels"].size(0) == 1
