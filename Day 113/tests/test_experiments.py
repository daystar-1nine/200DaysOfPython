import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import tempfile
from app.config import ModelConfig, TrainingConfig, SCALING_MODELS, load_config_from_yaml, save_config_to_yaml
from app.data.loader import load_raw_text, split_into_documents, build_character_vocab
from app.analysis.compute import (
    estimate_training_flops,
    estimate_chinchilla_optimal,
    tokens_per_second,
    estimated_training_time,
)
from app.analysis.params import (
    estimate_parameters,
    parameter_memory,
    estimate_training_memory,
    format_bytes,
)
from app.analysis.scaling import (
    power_law_loss,
    fit_power_law,
    predict_scaled_loss,
    analyze_scaling_efficiency,
)


class TestDataLoaderAndVocabulary:
    def test_split_into_documents_custom_delimiters(self):
        raw = "Doc 1 line 1\nDoc 1 line 2\n\nDoc 2 line 1\nDoc 2 line 2"
        docs = split_into_documents(raw, delimiter="\n\n")
        assert len(docs) == 2
        assert "Doc 1 line 1" in docs[0]
        assert "Doc 2 line 1" in docs[1]

    def test_split_into_documents_empty_corpus(self):
        assert split_into_documents("") == []
        assert split_into_documents("   \n\n   ") == []

    def test_build_character_vocab_completeness(self):
        text = "Hello, LLM scaling!"
        vocab, char2idx, idx2char = build_character_vocab(text)
        assert "<unk>" in vocab
        for char in text:
            assert char in char2idx
            idx = char2idx[char]
            assert idx2char[idx] == char

    def test_build_character_vocab_sorted_indices(self):
        text = "cba"
        vocab, char2idx, idx2char = build_character_vocab(text)
        assert vocab[0] == "<unk>"
        assert vocab[1] == "a"
        assert vocab[2] == "b"
        assert vocab[3] == "c"

    def test_build_character_vocab_length(self):
        text = "abc"
        vocab, char2idx, idx2char = build_character_vocab(text)
        assert len(vocab) == 4  # 3 chars + <unk>


class TestConfigAndSerialization:
    def test_model_config_default_values(self):
        cfg = ModelConfig()
        assert cfg.name == "Tiny"
        assert cfg.vocab_size == 64
        assert cfg.context_length == 64
        assert cfg.embed_dim == 128
        assert cfg.num_heads == 4
        assert cfg.num_layers == 4
        assert cfg.tie_weights is True

    def test_training_config_default_values(self):
        t_cfg = TrainingConfig()
        assert t_cfg.batch_size == 32
        assert t_cfg.gradient_accumulation_steps == 1
        assert t_cfg.scheduler_type == "cosine"
        assert t_cfg.grad_clip == 1.0
        assert t_cfg.learning_rate == 5e-4

    def test_scaling_models_registry(self):
        assert "tiny" in SCALING_MODELS
        assert "small" in SCALING_MODELS
        assert "medium" in SCALING_MODELS
        assert SCALING_MODELS["tiny"].embed_dim == 128
        assert SCALING_MODELS["small"].embed_dim == 256
        assert SCALING_MODELS["medium"].embed_dim == 384

    def test_config_as_dict(self):
        cfg = ModelConfig(name="Custom", embed_dim=128)
        d = cfg.__dict__
        assert d["name"] == "Custom"
        assert d["embed_dim"] == 128


class TestMetricFormulasAndScaling:
    def test_perplexity_calculation_exact(self):
        loss = 2.0
        ppl = math.exp(loss)
        assert ppl == pytest.approx(7.389056, abs=1e-5)

    def test_perplexity_zero_loss(self):
        assert math.exp(0.0) == 1.0

    def test_throughput_metric_formula(self):
        tokens = 100000
        seconds = 20.0
        tps = tokens_per_second(tokens, seconds)
        assert tps == 5000.0

    def test_throughput_zero_seconds_safeguard(self):
        assert tokens_per_second(1000, 0.0) == 0.0

    def test_estimated_training_time_precision(self):
        flops = 1e15
        hw_tflops = 100.0
        result = estimated_training_time(flops, hardware_tflops=hw_tflops, num_devices=1, model_flops_utilization=0.5)
        # achieved = 100 * 1e12 * 0.5 = 5e13 FLOPs/s. total_seconds = 1e15 / 5e13 = 20.0s
        assert result["estimated_seconds"] == 20.0
        assert result["hardware_peak_tflops_total"] == 100.0

    def test_estimated_training_time_invalid_hardware_raises(self):
        with pytest.raises(ValueError):
            estimated_training_time(1e12, hardware_tflops=-1.0)

    def test_precision_matrix_monotonicity(self):
        fp32_mem = parameter_memory(1_000_000, "fp32")
        fp16_mem = parameter_memory(1_000_000, "fp16")
        int8_mem = parameter_memory(1_000_000, "int8")
        int4_mem = parameter_memory(1_000_000, "int4")

        assert fp32_mem["bytes"] == 2 * fp16_mem["bytes"]
        assert fp16_mem["bytes"] == 2 * int8_mem["bytes"]
        assert int8_mem["bytes"] == 2 * int4_mem["bytes"]

    def test_format_bytes_gigabytes_and_terabytes(self):
        gb_val = 2 * (1024 ** 3)
        tb_val = 5 * (1024 ** 4)
        assert "GB" in format_bytes(gb_val)
        assert "TB" in format_bytes(tb_val)

    def test_chinchilla_scaling_proportionality(self):
        c1 = 1e18
        c2 = 1e20
        res1 = estimate_chinchilla_optimal(c1)
        res2 = estimate_chinchilla_optimal(c2)
        ratio_n = res2["optimal_parameters"] / res1["optimal_parameters"]
        ratio_d = res2["optimal_tokens"] / res1["optimal_tokens"]
        assert 9.0 <= ratio_n <= 11.0
        assert 9.0 <= ratio_d <= 11.0

    def test_training_memory_adamw_overhead(self):
        mem = estimate_training_memory(
            parameters=100_000_000,
            batch_size=8,
            context_length=1024,
            num_layers=12,
            embed_dim=768,
            precision="fp32"
        )
        assert mem["optimizer_memory_gb"] > 0
        assert mem["gradients_memory_gb"] > 0
        assert mem["total_training_memory_gb"] > mem["model_memory_gb"] + mem["optimizer_memory_gb"]

    def test_training_flops_heuristic(self):
        flops = estimate_training_flops(parameters=1_000_000, tokens=20_000_000)
        assert flops["total_training_flops"] == 6 * 1_000_000 * 20_000_000
        assert flops["forward_flops"] == 2 * 1_000_000 * 20_000_000
        assert flops["backward_flops"] == 4 * 1_000_000 * 20_000_000

    def test_fit_power_law_insufficient_points_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            fit_power_law([1e6], [3.2])

    def test_fit_power_law_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            fit_power_law([1e6, 2e6], [3.2])

    def test_analyze_scaling_efficiency_single_model(self):
        res = analyze_scaling_efficiency([1_000_000], [3.0])
        assert len(res) == 1
        assert res[0]["marginal_improvement"] is None

    def test_power_law_loss_positive_output(self):
        loss = power_law_loss(1e7, a=10.0, alpha=0.3, irreducible_loss=1.6)
        assert loss > 1.6

    def test_load_raw_text_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_raw_text("missing_corpus_file.txt")

    def test_load_raw_text_valid_file(self):
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write("Pretraining LLMs requires data!")
            f_name = f.name
        try:
            content = load_raw_text(f_name)
            assert content == "Pretraining LLMs requires data!"
        finally:
            Path(f_name).unlink(missing_ok=True)

    def test_save_and_load_yaml_roundtrip(self):
        cfg = {"model": "TinyGPT", "vocab_size": 128, "layers": 4}
        with tempfile.TemporaryDirectory() as tmpdir:
            yml_path = Path(tmpdir) / "sub" / "cfg.yaml"
            save_config_to_yaml(cfg, yml_path)
            assert yml_path.exists()
            loaded = load_config_from_yaml(yml_path)
            assert loaded == cfg

    def test_estimate_parameters_ff_multiplier(self):
        p4 = estimate_parameters(vocab_size=100, embed_dim=64, num_layers=2, num_heads=2, ff_multiplier=4)
        p8 = estimate_parameters(vocab_size=100, embed_dim=64, num_layers=2, num_heads=2, ff_multiplier=8)
        assert p8["mlp_per_block"] > p4["mlp_per_block"]

    def test_estimate_parameters_with_and_without_biases(self):
        p_bias = estimate_parameters(vocab_size=100, embed_dim=64, num_layers=2, num_heads=2, include_biases=True)
        p_nobias = estimate_parameters(vocab_size=100, embed_dim=64, num_layers=2, num_heads=2, include_biases=False)
        assert p_bias["total_parameters"] > p_nobias["total_parameters"]

    def test_format_bytes_petabytes(self):
        pb_val = 3.5 * (1024 ** 5)
        formatted = format_bytes(pb_val)
        assert "PB" in formatted

