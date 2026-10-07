"""
Unit tests for PreferenceCollator: pairwise tokenization, dynamic padding,
and assistant-only loss masking for DPO and Reward Modeling.
"""
import pytest
import torch

from app.data.collator import PreferenceCollator


def test_collator_initialization(dummy_tokenizer):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer, max_length=64, ignore_index=-100)
    assert collator.tokenizer == dummy_tokenizer
    assert collator.max_length == 64
    assert collator.ignore_index == -100


def test_collator_output_keys(dummy_tokenizer, sample_preference_batch):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator(sample_preference_batch)

    expected_keys = {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
        "prompt_lengths"
    }
    assert set(batch.keys()) == expected_keys


def test_collator_tensor_types(dummy_tokenizer, sample_preference_batch):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator(sample_preference_batch)

    for k, v in batch.items():
        assert isinstance(v, torch.Tensor)
        assert v.dtype == torch.long


def test_collator_tensor_shapes(dummy_tokenizer, sample_preference_batch):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator(sample_preference_batch)

    B = len(sample_preference_batch)
    assert batch["chosen_input_ids"].size(0) == B
    assert batch["chosen_attention_mask"].size(0) == B
    assert batch["chosen_labels"].size(0) == B
    assert batch["rejected_input_ids"].size(0) == B
    assert batch["rejected_attention_mask"].size(0) == B
    assert batch["rejected_labels"].size(0) == B
    assert batch["prompt_lengths"].size(0) == B


def test_prompt_tokens_masked_in_chosen_labels(dummy_tokenizer, sample_preference_pair):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator([sample_preference_pair])

    c_labels = batch["chosen_labels"][0]
    prompt_len = batch["prompt_lengths"][0].item()

    # All tokens from 0 to prompt_len - 1 must be -100
    for idx in range(prompt_len):
        assert c_labels[idx].item() == -100, f"Token at pos {idx} should be masked to -100"


def test_chosen_response_tokens_unmasked(dummy_tokenizer, sample_preference_pair):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator([sample_preference_pair])

    c_labels = batch["chosen_labels"][0]
    prompt_len = batch["prompt_lengths"][0].item()
    c_ids = batch["chosen_input_ids"][0]

    # Non-padding response tokens should match c_ids
    non_pad_mask = batch["chosen_attention_mask"][0] == 1
    response_indices = [i for i in range(prompt_len, len(c_labels)) if non_pad_mask[i].item() == 1]

    assert len(response_indices) > 0
    for idx in response_indices:
        assert c_labels[idx].item() == c_ids[idx].item()


def test_prompt_tokens_masked_in_rejected_labels(dummy_tokenizer, sample_preference_pair):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator([sample_preference_pair])

    r_labels = batch["rejected_labels"][0]
    prompt_len = batch["prompt_lengths"][0].item()

    for idx in range(prompt_len):
        assert r_labels[idx].item() == -100, f"Rejected token at pos {idx} should be masked to -100"


def test_rejected_response_tokens_unmasked(dummy_tokenizer, sample_preference_pair):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator([sample_preference_pair])

    r_labels = batch["rejected_labels"][0]
    prompt_len = batch["prompt_lengths"][0].item()
    r_ids = batch["rejected_input_ids"][0]

    non_pad_mask = batch["rejected_attention_mask"][0] == 1
    response_indices = [i for i in range(prompt_len, len(r_labels)) if non_pad_mask[i].item() == 1]

    assert len(response_indices) > 0
    for idx in response_indices:
        assert r_labels[idx].item() == r_ids[idx].item()


def test_end_token_unmasked_at_end_of_response(dummy_tokenizer, sample_preference_pair):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator([sample_preference_pair])

    c_labels = batch["chosen_labels"][0]
    c_ids = batch["chosen_input_ids"][0]
    end_id = dummy_tokenizer.end_id

    # Find position of last non-pad token in chosen
    non_pad = (batch["chosen_attention_mask"][0] == 1).nonzero(as_tuple=True)[0]
    last_idx = non_pad[-1].item()

    assert c_ids[last_idx].item() == end_id
    assert c_labels[last_idx].item() == end_id


def test_dynamic_padding_and_attention_mask(dummy_tokenizer):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    items = [
        {"prompt": "Q1", "chosen": "a", "rejected": "b"},
        {"prompt": "Q2", "chosen": "a b c d e", "rejected": "b c d e"}
    ]
    batch = collator(items)

    pad_id = dummy_tokenizer.pad_id
    short_input = batch["chosen_input_ids"][0]
    short_mask = batch["chosen_attention_mask"][0]
    short_labels = batch["chosen_labels"][0]

    pad_indices = (short_input == pad_id).nonzero(as_tuple=True)[0]
    assert len(pad_indices) > 0

    for idx in pad_indices:
        assert short_mask[idx].item() == 0, "Pad position must have attention mask 0"
        assert short_labels[idx].item() == -100, "Pad position must have label -100"


def test_truncation_when_exceeding_max_length(dummy_tokenizer):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer, max_length=15)
    item = {
        "prompt": "Explain very long prompt that exceeds length limit",
        "chosen": "Long chosen response with many tokens",
        "rejected": "Long rejected response with many tokens"
    }
    batch = collator([item])
    assert batch["chosen_input_ids"].size(1) <= 15
    assert batch["rejected_input_ids"].size(1) <= 15


def test_single_item_batch(dummy_tokenizer, sample_preference_pair):
    collator = PreferenceCollator(tokenizer=dummy_tokenizer)
    batch = collator([sample_preference_pair])
    assert batch["chosen_input_ids"].size(0) == 1
    assert batch["rejected_input_ids"].size(0) == 1
    assert batch["prompt_lengths"].size(0) == 1
