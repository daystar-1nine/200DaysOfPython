"""
Unit tests for interactive chat, inference, evaluation protocols,
and end-to-end multi-turn conversation dynamics for MiniGPT-Chat.
"""
import pytest
import json
import torch
from pathlib import Path

from app.config import ChatModelConfig, GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import format_chat, extract_assistant_response
from app.model.minigpt_chat import MiniGPTChat
from app.inference.chat import ChatSession
from app.evaluation.evaluator import score_response, evaluate_model, compare_base_vs_sft, save_evaluation_results
from app.training.losses import compute_masked_loss, compute_token_accuracy
from app.training.checkpoint import save_sft_checkpoint, load_sft_checkpoint


@pytest.fixture
def tokenizer():
    vocab = {
        "<|pad|>": 0, "<|system|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|end|>": 4,
        "a": 5, "b": 6, "c": 7, "d": 8, " ": 9, "p": 10, "y": 11, "t": 12, "h": 13, "o": 14, "n": 15,
        "i": 16, "s": 17, "g": 18, "r": 19, "e": 20, "a": 21, "!": 22, "?": 23, ".": 24
    }
    tok = ChatTokenizer()
    tok.vocab = vocab
    tok.inverse_vocab = {v: k for k, v in vocab.items()}
    return tok


@pytest.fixture
def model(tokenizer):
    cfg = ChatModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        embed_dim=32,
        num_heads=2,
        num_layers=2
    )
    return MiniGPTChat(cfg)


def test_format_chat_generation_prompt_suffix():
    msgs = [{"role": "user", "content": "hello"}]
    formatted = format_chat(msgs, add_generation_prompt=True)
    assert formatted.endswith("<|assistant|>")


def test_extract_assistant_response_with_end_token():
    raw = "<|system|>Sys<|end|><|user|>Hi<|end|><|assistant|>Hello world!<|end|>"
    res = extract_assistant_response(raw)
    assert res == "Hello world!"


def test_extract_assistant_response_without_end_token():
    raw = "<|system|>Sys<|end|><|user|>Hi<|end|><|assistant|>Partial reply"
    res = extract_assistant_response(raw)
    assert res == "Partial reply"


def test_extract_assistant_response_no_assistant_tag():
    raw = "Direct plain text response"
    res = extract_assistant_response(raw)
    assert res == "Direct plain text response"


def test_chat_session_system_prompt_preserved(model, tokenizer):
    session = ChatSession(model, tokenizer, system_prompt="Custom System Prompt")
    assert session.messages[0]["content"] == "Custom System Prompt"
    session.add_message("user", "Hello")
    assert len(session.messages) == 2


def test_chat_session_reset(model, tokenizer):
    session = ChatSession(model, tokenizer, system_prompt="Helper")
    session.add_message("user", "Question 1")
    session.add_message("assistant", "Answer 1")
    session.reset()
    assert len(session.messages) == 1
    assert session.messages[0]["content"] == "Helper"


def test_chat_session_history_truncation(model, tokenizer):
    session = ChatSession(model, tokenizer, system_prompt="Short")
    for i in range(8):
        session.add_message("user", f"query {i}")
        session.add_message("assistant", f"reply {i}")

    initial_len = len(session.messages)
    session.truncate_history(max_tokens=25)
    # Truncated messages should be fewer, but system prompt still at index 0
    assert len(session.messages) < initial_len
    assert session.messages[0]["role"] == "system"


def test_greedy_decoding_reproducibility(model, tokenizer):
    prompt_ids = torch.tensor([[2, 5, 1, 3, 6, 1, 4]], dtype=torch.long)
    out1 = model.generate(prompt_ids, max_new_tokens=6, do_sample=False)
    out2 = model.generate(prompt_ids, max_new_tokens=6, do_sample=False)
    assert torch.equal(out1, out2)


def test_rubric_score_0_on_gibberish():
    score, rationale = score_response("zzzzzzzzzzzzzzzzzzz", "Photosynthesis is light energy", "What is it?")
    assert score == 0


def test_rubric_score_1_on_partial_keyword_match():
    score, rationale = score_response("It relates to energy conversion in nature.", "Photosynthesis converts light energy into sugar.", "What is photosynthesis?")
    assert score == 1
    assert "Partially followed" in rationale


def test_rubric_score_2_on_good_match():
    score, rationale = score_response(
        "Photosynthesis converts light energy into chemical energy in plants.",
        "Photosynthesis is the process that converts light energy to chemical energy.",
        "Explain photosynthesis"
    )
    assert score == 2
    assert "Correctly followed" in rationale


def test_evaluate_model_empty_test_set(model, tokenizer):
    res = evaluate_model(model, [], tokenizer)
    assert res["overall_score_pct"] == 0.0
    assert res["evaluated_prompts"] == 0


def test_evaluate_model_category_breakdown(model, tokenizer):
    test_set = [
        {"messages": [{"role": "user", "content": "p1"}, {"role": "assistant", "content": "a1"}], "category": "cat_a"},
        {"messages": [{"role": "user", "content": "p2"}, {"role": "assistant", "content": "a2"}], "category": "cat_b"}
    ]
    res = evaluate_model(model, test_set, tokenizer)
    assert "cat_a" in res["category_breakdown"]
    assert "cat_b" in res["category_breakdown"]
    assert res["category_breakdown"]["cat_a"]["count"] == 1
    assert res["category_breakdown"]["cat_b"]["count"] == 1


def test_save_and_load_evaluation_json(tmp_path):
    data = {"overall_score_pct": 85.5, "total_points": 171, "max_points": 200}
    filepath = tmp_path / "eval.json"
    save_evaluation_results(data, filepath)
    assert filepath.exists()

    with open(filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["overall_score_pct"] == 85.5
    assert loaded["total_points"] == 171


def test_compare_base_vs_sft_structure(model, tokenizer):
    items = [{"instruction": "Explain recursion.", "ground_truth": "A function that calls itself."}]
    res = compare_base_vs_sft(model, model, items, tokenizer)
    assert len(res) == 1
    assert "instruction" in res[0] or "prompt" in res[0]
    assert "improvement" in res[0]


def test_checkpoint_saves_metadata(tmp_path, model):
    ckpt_file = tmp_path / "meta_test.pt"
    save_sft_checkpoint(
        ckpt_file,
        model,
        epoch=3,
        step=150,
        metrics={"val_loss": 2.15, "val_ppl": 8.58},
        config={"lr": 3e-4}
    )
    loaded = load_sft_checkpoint(ckpt_file, model)
    assert loaded["epoch"] == 3
    assert loaded["step"] == 150
    assert loaded["metrics"]["val_loss"] == 2.15
    assert loaded["config"]["lr"] == 3e-4


def test_masked_loss_when_all_targets_ignored():
    logits = torch.randn(2, 6, 10)
    targets = torch.full((2, 6), -100, dtype=torch.long)
    loss, ppl, active = compute_masked_loss(logits, targets)
    assert active == 0
    assert ppl == 1.0
    assert loss.item() == 0.0


def test_token_accuracy_with_all_matching():
    logits = torch.zeros(1, 4, 5)
    logits[0, :, 3] = 10.0  # Class 3 has highest probability everywhere
    # shift_targets will be [3, 3, 3]
    targets = torch.tensor([[-100, 3, 3, 3]], dtype=torch.long)
    acc = compute_token_accuracy(logits, targets)
    assert acc == 1.0


def test_special_tokens_property_indices(tokenizer):
    assert tokenizer.pad_id == 0
    assert tokenizer.system_id == 1
    assert tokenizer.user_id == 2
    assert tokenizer.assistant_id == 3
    assert tokenizer.end_id == 4


def test_generation_config_repetition_penalty_applied(model):
    # Verify repetition penalty runs without exception
    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    gen = model.generate(prompt_ids, max_new_tokens=5, repetition_penalty=1.5, do_sample=False)
    assert gen.size(1) == 8
