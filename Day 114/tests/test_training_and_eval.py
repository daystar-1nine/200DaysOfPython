"""
Unit tests for SFT training engine, loss masking, accuracy metrics,
learning rate schedules, checkpoint management, evaluation rubric scoring,
and interactive multi-turn chat session.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from app.config import ChatModelConfig, SFTTrainingConfig, GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.model.minigpt_chat import MiniGPTChat
from app.training.losses import compute_masked_loss, compute_token_accuracy
from app.training.trainer import get_cosine_schedule_with_warmup, SFTTrainer
from app.training.checkpoint import save_sft_checkpoint, load_sft_checkpoint
from app.evaluation.evaluator import score_response, evaluate_model, compare_base_vs_sft
from app.inference.chat import ChatSession


@pytest.fixture
def dummy_tokenizer():
    tok = ChatTokenizer()
    tok.vocab = {
        "<|pad|>": 0, "<|end|>": 1, "<|system|>": 2, "<|user|>": 3, "<|assistant|>": 4,
        "a": 5, "b": 6, "c": 7, "d": 8, " ": 9, "p": 10, "y": 11, "t": 12, "h": 13, "o": 14, "n": 15
    }
    tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
    return tok


@pytest.fixture
def mini_model(dummy_tokenizer):
    cfg = ChatModelConfig(
        vocab_size=dummy_tokenizer.vocab_size,
        context_length=32,
        embed_dim=32,
        num_heads=2,
        num_layers=2
    )
    return MiniGPTChat(cfg)


def test_compute_masked_loss_calculation():
    # [B, T, V] = [1, 4, 5]
    logits = torch.zeros(1, 4, 5)
    # shift_targets is targets[:, 1:], so targets shape [1, 4]
    # Pos 0: prompt (ignored in shift), pos 1: -100, pos 2: -100, pos 3: class 0
    targets = torch.tensor([[ -100, -100, -100, 0 ]], dtype=torch.long)
    # When logits are all 0, softmax prob for any class is 1/5 = 0.2
    # -ln(0.2) = ln(5) = 1.6094
    loss, ppl, active = compute_masked_loss(logits, targets, ignore_index=-100)
    expected = -torch.log(torch.tensor(0.2))
    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)
    assert active == 1
    assert ppl > 1.0


def test_token_accuracy_metric():
    # [B, T, V] = [1, 5, 3]
    logits = torch.zeros(1, 5, 3)
    # Class 0 has highest logit for all positions
    logits[:, :, 0] = 5.0

    # targets[:, 1:] will be [0, -100, 1, 0]
    targets = torch.tensor([[-100, 0, -100, 1, 0]], dtype=torch.long)
    # Valid positions in shift_targets:
    # pos 0 (target 0, pred 0 -> match), pos 2 (target 1, pred 0 -> mismatch), pos 3 (target 0, pred 0 -> match)
    # Accuracy should be 2/3 = ~0.6667
    acc = compute_token_accuracy(logits, targets, ignore_index=-100)
    assert round(acc, 3) == 0.667


def test_token_accuracy_all_masked():
    logits = torch.randn(1, 5, 3)
    targets = torch.full((1, 5), -100, dtype=torch.long)
    acc = compute_token_accuracy(logits, targets, ignore_index=-100)
    assert acc == 0.0


def test_cosine_warmup_scheduler():
    model = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100, min_lr_ratio=0.1)

    # Step 0: Warmup start
    assert optimizer.param_groups[0]["lr"] == 0.0

    # Step 10: Warmup end (should reach peak lr 1.0)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    torch.testing.assert_close(torch.tensor(optimizer.param_groups[0]["lr"]), torch.tensor(1.0), atol=1e-3, rtol=1e-3)

    # Step 100: Final step (should reach min_lr 0.1)
    for _ in range(90):
        optimizer.step()
        scheduler.step()
    torch.testing.assert_close(torch.tensor(optimizer.param_groups[0]["lr"]), torch.tensor(0.1), atol=1e-3, rtol=1e-3)


def test_checkpoint_save_and_load(tmp_path, mini_model):
    ckpt_path = tmp_path / "model_ckpt.pt"
    optimizer = torch.optim.AdamW(mini_model.parameters(), lr=1e-3)

    # Save checkpoint
    save_sft_checkpoint(
        ckpt_path,
        mini_model,
        optimizer=optimizer,
        epoch=2,
        step=42,
        metrics={"val_loss": 1.234}
    )
    assert ckpt_path.exists()

    # Load checkpoint into new model
    cfg = mini_model.config
    new_model = MiniGPTChat(cfg)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

    meta = load_sft_checkpoint(ckpt_path, new_model, optimizer=new_opt)
    assert meta["step"] == 42
    assert meta["epoch"] == 2
    assert meta["metrics"]["val_loss"] == 1.234

    # Verify weights match
    for p1, p2 in zip(mini_model.parameters(), new_model.parameters()):
        torch.testing.assert_close(p1, p2)


def test_rubric_scoring_correct():
    gen = "Python is a high-level programming language used widely in data science."
    gt = "Python is a high-level programming language."
    prompt = "What is Python?"
    score, rationale = score_response(gen, gt, prompt)
    assert score == 2
    assert "Correctly followed" in rationale


def test_rubric_scoring_empty():
    score, rationale = score_response("", "Some ground truth", "What is Python?")
    assert score == 0
    assert "empty or too short" in rationale


def test_rubric_scoring_repetition():
    gen = "python python python python python python python python"
    score, rationale = score_response(gen, "Python is a programming language", "What is Python?")
    assert score == 0
    assert "Degenerate" in rationale


def test_chat_session_lifecycle(mini_model, dummy_tokenizer):
    session = ChatSession(
        model=mini_model,
        tokenizer=dummy_tokenizer,
        system_prompt="You are a helper."
    )
    # Starts with system message
    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "system"

    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there")
    assert len(session.messages) == 3

    # Add multiple turns and truncate history
    for i in range(10):
        session.add_message("user", f"Question {i}")
        session.add_message("assistant", f"Answer {i}")

    assert len(session.messages) > 5
    session.truncate_history(max_tokens=30)
    # System message must always be preserved
    assert session.messages[0]["role"] == "system"
    assert session.messages[0]["content"] == "You are a helper."

    session.reset()
    assert len(session.messages) == 1
    assert session.messages[0]["content"] == "You are a helper."


def test_evaluator_scoring_aggregation(mini_model, dummy_tokenizer):
    test_data = [
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is language."}
            ],
            "category": "python"
        }
    ]
    res = evaluate_model(mini_model, test_data, dummy_tokenizer)
    assert "overall_score_pct" in res
    assert "category_breakdown" in res
    assert "results" in res
    assert len(res["results"]) == 1


def test_compare_base_vs_sft_output_schema(mini_model, dummy_tokenizer):
    sft_model = mini_model
    prompts = [{"instruction": "What is a list?", "ground_truth": "A list is ordered."}]
    comps = compare_base_vs_sft(mini_model, sft_model, prompts, dummy_tokenizer)
    assert len(comps) == 1
    assert "prompt" in comps[0]
    assert "base_response" in comps[0]
    assert "sft_response" in comps[0]
    assert "improvement" in comps[0]
