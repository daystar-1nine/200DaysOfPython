"""
Unit tests for evaluation metrics, quality rubrics, interactive chat session, and tokenizer formatting.
"""
import pytest
import torch

from app.config import ModelConfig, RewardModelConfig, DPOTrainingConfig, GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import (
    format_prompt,
    format_chat_prompt,
    format_preference_pair,
    extract_assistant_response,
    validate_preference_pair
)
from app.models.minigpt import MiniGPTBackbone
from app.models.dpo_model import MiniGPTChat, DPOModel
from app.models.reward_model import RewardModel
from app.evaluation.evaluator import score_response_quality, compare_models_on_prompts
from app.inference.chat import PreferenceChatSession


def test_score_response_quality_empty():
    res = score_response_quality("", "A list is a sequence.", "What is a list?")
    assert res["instruction_following"] == 0.0
    assert res["helpfulness"] == 0.0
    assert res["conciseness"] == 0.0
    assert res["composite"] == 0.0


def test_score_response_quality_repetitive_loop_penalized():
    loop_text = "the the the the the the the the the the"
    res = score_response_quality(loop_text, "A list is a sequence.", "What is a list?")
    assert res["composite"] == 0.0


def test_score_response_quality_high_overlap():
    prompt = "What is a python list?"
    gt = "A list in Python is an ordered, mutable sequence of elements."
    pred = "A list in Python is an ordered, mutable sequence of elements."
    res = score_response_quality(pred, gt, prompt)
    assert res["instruction_following"] == 100.0
    assert res["helpfulness"] == 100.0
    assert res["conciseness"] == 100.0
    assert res["composite"] == 100.0


def test_score_response_quality_conciseness_penalty_on_long_text():
    prompt = "What is gravity?"
    gt = "Gravity pulls objects toward each other."
    pred = "Gravity is a fundamental interaction " * 15  # 75 words
    res = score_response_quality(pred, gt, prompt)
    assert res["conciseness"] <= 30.0


def test_compare_models_on_prompts(small_model_config, dummy_tokenizer):
    base_model = MiniGPTChat(small_model_config)
    dpo_model = MiniGPTChat(small_model_config)
    models = {"Base": base_model, "DPO": dpo_model}

    test_prompts = [
        {"prompt": "Say hi", "chosen": "Hello!"},
        {"prompt": "What is 2+2?", "chosen": "4"}
    ]
    gen_cfg = GenerationConfig(max_new_tokens=4, do_sample=False)

    itemized, agg = compare_models_on_prompts(
        models=models,
        test_prompts=test_prompts,
        tokenizer=dummy_tokenizer,
        generation_config=gen_cfg
    )

    assert len(itemized) == 2
    assert "Base" in agg
    assert "DPO" in agg
    assert "composite_score" in agg["DPO"]
    assert "avg_response_length" in agg["Base"]


def test_preference_chat_session_initialization(small_model_config, dummy_tokenizer):
    dpo = MiniGPTChat(small_model_config)
    session = PreferenceChatSession(dpo_model=dpo, tokenizer=dummy_tokenizer, system_prompt="Test Assistant")
    assert session.system_prompt == "Test Assistant"
    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "system"


def test_preference_chat_session_add_message_and_reset(small_model_config, dummy_tokenizer):
    dpo = MiniGPTChat(small_model_config)
    session = PreferenceChatSession(dpo_model=dpo, tokenizer=dummy_tokenizer)
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there!")
    assert len(session.messages) == 3

    session.reset()
    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "system"


def test_preference_chat_session_sliding_window_truncation(small_model_config, dummy_tokenizer):
    dpo = MiniGPTChat(small_model_config)
    session = PreferenceChatSession(dpo_model=dpo, tokenizer=dummy_tokenizer)

    for i in range(12):
        session.add_message("user", f"Turn {i} question")
        session.add_message("assistant", f"Turn {i} answer")

    session.truncate_history(max_tokens=25)
    # System prompt remains intact
    assert session.messages[0]["role"] == "system"
    # Older turns evicted
    assert len(session.messages) < 25


def test_preference_chat_session_generate_reply(small_model_config, dummy_tokenizer):
    dpo = MiniGPTChat(small_model_config)
    session = PreferenceChatSession(dpo_model=dpo, tokenizer=dummy_tokenizer)
    reply = session.generate_reply("Hello!")
    assert isinstance(reply, str)
    assert len(session.messages) == 3  # system, user, assistant


def test_preference_chat_session_compare_turn(small_model_config, dummy_tokenizer):
    dpo = MiniGPTChat(small_model_config)
    sft = MiniGPTChat(small_model_config)
    rm = RewardModel(model_config=small_model_config)

    session = PreferenceChatSession(
        dpo_model=dpo,
        sft_model=sft,
        reward_model=rm,
        tokenizer=dummy_tokenizer
    )

    cmp_res = session.compare_turn("What is 1+1?")
    assert "dpo_response" in cmp_res
    assert "sft_response" in cmp_res
    assert "dpo_reward" in cmp_res
    assert "sft_reward" in cmp_res


def test_tokenizer_special_tokens():
    tok = ChatTokenizer()
    assert tok.pad_id == 0
    assert tok.system_id == 1
    assert tok.user_id == 2
    assert tok.assistant_id == 3
    assert tok.end_id == 4
    assert tok.unk_id == 5


def test_tokenizer_encode_decode():
    tok = ChatTokenizer()
    text = "Hello world! 123"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)


def test_tokenizer_handles_unknown_tokens():
    tok = ChatTokenizer()
    ids = tok.encode("\U0001f600 \u2603")
    assert tok.unk_id in ids


def test_format_chat_prompt():
    msgs = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "A language."}
    ]
    prompt_str = format_chat_prompt(msgs, system_prompt="Be concise.")
    assert "<|system|>Be concise.<|end|>" in prompt_str
    assert "<|user|>What is Python?<|end|>" in prompt_str
    assert "<|assistant|>A language.<|end|>" in prompt_str
    assert prompt_str.endswith("<|assistant|>")


def test_format_preference_pair():
    p, c, r = format_preference_pair("Prompt", "Chosen", "Rejected")
    assert "<|user|>Prompt<|end|><|assistant|>" in p
    assert c == f"{p}Chosen<|end|>"
    assert r == f"{p}Rejected<|end|>"


def test_validate_preference_pair_logic():
    assert validate_preference_pair({"prompt": "p", "chosen": "c", "rejected": "r"})
    assert not validate_preference_pair({"prompt": "p", "chosen": "c", "rejected": "c"})
    assert not validate_preference_pair({"prompt": "", "chosen": "c", "rejected": "r"})
    assert not validate_preference_pair("not a dict")


def test_extract_assistant_response():
    raw = "<|system|>Sys<|end|><|user|>Hi<|end|><|assistant|>Hello there!<|end|>"
    assert extract_assistant_response(raw) == "Hello there!"


def test_generation_config_attributes():
    cfg = GenerationConfig(max_new_tokens=20, temperature=0.5, do_sample=False)
    assert cfg.max_new_tokens == 20
    assert cfg.temperature == 0.5
    assert not cfg.do_sample


def test_minigpt_chat_forward_loss(small_model_config):
    model = MiniGPTChat(small_model_config)
    ids = torch.randint(0, small_model_config.vocab_size, (2, 8))
    labels = ids.clone()
    logits, loss = model(ids, labels=labels)
    assert logits.shape == (2, 8, small_model_config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0


def test_minigpt_chat_forward_inference(small_model_config):
    model = MiniGPTChat(small_model_config)
    ids = torch.randint(0, small_model_config.vocab_size, (1, 6))
    logits, loss = model(ids)
    assert logits.shape == (1, 6, small_model_config.vocab_size)
    assert loss is None


def test_minigpt_backbone_residual_connections(small_model_config):
    backbone = MiniGPTBackbone(small_model_config)
    assert len(backbone.blocks) == small_model_config.num_layers
    ids = torch.randint(0, small_model_config.vocab_size, (2, 4))
    h = backbone(ids)
    assert h.shape == (2, 4, small_model_config.embed_dim)


def test_reward_model_pooling_methods(small_model_config):
    cfg_last = RewardModelConfig(pooling_method="last")
    cfg_mean = RewardModelConfig(pooling_method="mean")
    rm_last = RewardModel(model_config=small_model_config, reward_config=cfg_last)
    rm_mean = RewardModel(model_config=small_model_config, reward_config=cfg_mean)

    ids = torch.randint(0, small_model_config.vocab_size, (2, 6))
    mask = torch.ones_like(ids)

    r_l = rm_last(ids, mask)
    r_m = rm_mean(ids, mask)
    assert r_l.shape == (2,)
    assert r_m.shape == (2,)


def test_dpo_config_loss_types():
    for lt in ["sigmoid", "hinge", "ipo"]:
        cfg = DPOTrainingConfig(loss_type=lt)
        assert cfg.loss_type == lt


def test_evaluator_empty_prompts_list(small_model_config, dummy_tokenizer):
    base_model = MiniGPTChat(small_model_config)
    models = {"Base": base_model}
    itemized, agg = compare_models_on_prompts(models, [], dummy_tokenizer)
    assert len(itemized) == 0
    assert "Base" in agg


def test_format_prompt_without_generation_prompt():
    prompt_str = format_prompt("Hello", system_prompt="Sys", add_generation_prompt=False)
    assert prompt_str.endswith("<|end|>")
    assert not prompt_str.endswith("<|assistant|>")


def test_preference_chat_session_history_preserves_order(small_model_config, dummy_tokenizer):
    dpo = MiniGPTChat(small_model_config)
    session = PreferenceChatSession(dpo_model=dpo, tokenizer=dummy_tokenizer)
    session.add_message("user", "Msg 1")
    session.add_message("assistant", "Msg 2")
    session.add_message("user", "Msg 3")
    contents = [m["content"] for m in session.messages]
    assert contents == [session.system_prompt, "Msg 1", "Msg 2", "Msg 3"]


def test_dpo_model_eval_mode_disables_dropout(small_model_config):
    policy = MiniGPTChat(small_model_config)
    ref = MiniGPTChat(small_model_config)
    dpo = DPOModel(policy_model=policy, reference_model=ref)
    dpo.eval()
    assert not dpo.training
    assert not dpo.policy.training
    assert not dpo.reference.training


def test_reward_model_eval_mode_disables_dropout(small_model_config):
    rm = RewardModel(model_config=small_model_config)
    rm.eval()
    assert not rm.training
    assert not rm.backbone.training

