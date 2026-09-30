"""
Unit tests for Day 113 Analysis and Estimator Modules:
- Parameter counting and breakdown
- Static parameter memory across precisions (FP32, FP16, BF16, INT8, INT4)
- Dynamic training memory (model + grads + optimizer states + activations)
- FLOPs calculation (6ND heuristic)
- Chinchilla compute-optimal ratios
- Power-law scaling fitting and evaluation
"""
import math
import pytest
from app.analysis.params import (
    estimate_parameters,
    parameter_memory,
    estimate_training_memory,
    format_bytes
)
from app.analysis.compute import (
    estimate_training_flops,
    estimate_chinchilla_optimal,
    tokens_per_second,
    estimated_training_time
)
from app.analysis.scaling import (
    power_law_loss,
    fit_power_law,
    predict_scaled_loss,
    analyze_scaling_efficiency
)


class TestParameterEstimator:
    def test_estimate_parameters_tiny_breakdown(self):
        res = estimate_parameters(
            vocab_size=64,
            embed_dim=128,
            num_layers=4,
            num_heads=4,
            context_length=64,
            tie_weights=True
        )
        assert res["vocab_size"] == 64
        assert res["embed_dim"] == 128
        assert res["token_embeddings"] == 64 * 128
        assert res["position_embeddings"] == 64 * 128
        assert res["lm_head"] == 0  # Tied
        assert res["total_parameters"] > 800_000
        assert res["total_parameters"] < 820_000

    def test_estimate_parameters_untied_adds_lm_head(self):
        tied = estimate_parameters(vocab_size=1000, embed_dim=256, num_layers=4, num_heads=4, tie_weights=True)
        untied = estimate_parameters(vocab_size=1000, embed_dim=256, num_layers=4, num_heads=4, tie_weights=False)
        assert untied["lm_head"] == 1000 * 256
        assert untied["total_parameters"] == tied["total_parameters"] + (1000 * 256)

    def test_estimate_parameters_attention_and_mlp_formulas(self):
        d = 64
        res = estimate_parameters(vocab_size=32, embed_dim=d, num_layers=1, num_heads=2, include_biases=True)
        # Attention: 3*d^2 (QKV) + 3*d (bias) + d^2 (proj) + d (bias) = 4*d^2 + 4*d
        expected_attn = 4 * (d ** 2) + 4 * d
        assert res["attn_per_block"] == expected_attn
        # MLP: d*(4d) + 4d (bias) + (4d)*d + d (bias) = 8*d^2 + 5*d
        expected_mlp = 8 * (d ** 2) + 5 * d
        assert res["mlp_per_block"] == expected_mlp

    def test_parameter_memory_precision_byte_factors(self):
        n = 1_000_000
        fp32_mem = parameter_memory(n, precision="fp32")
        fp16_mem = parameter_memory(n, precision="fp16")
        bf16_mem = parameter_memory(n, precision="bf16")
        int8_mem = parameter_memory(n, precision="int8")
        int4_mem = parameter_memory(n, precision="int4")

        assert fp32_mem["bytes"] == 4_000_000
        assert fp16_mem["bytes"] == 2_000_000
        assert bf16_mem["bytes"] == 2_000_000
        assert int8_mem["bytes"] == 1_000_000
        assert int4_mem["bytes"] == 500_000

    def test_parameter_memory_gigabyte_conversion(self):
        # 1 billion parameters in FP32 ~= 4 GB
        mem = parameter_memory(1_000_000_000, precision="fp32")
        expected_gb = (1_000_000_000 * 4) / (1024 ** 3)
        assert math.isclose(mem["gigabytes"], expected_gb, rel_tol=1e-4)

    def test_estimate_training_memory_components(self):
        params = 10_000_000
        mem = estimate_training_memory(params, optimizer="adamw", precision="fp32")
        assert mem["model_memory_gb"] > 0
        assert mem["gradients_memory_gb"] > 0
        assert mem["optimizer_memory_gb"] > 0
        assert mem["activations_memory_gb"] > 0
        assert mem["total_training_memory_gb"] > mem["model_memory_gb"] * 3

    def test_format_bytes_scales_units(self):
        assert "B" in format_bytes(500)
        assert "KB" in format_bytes(2048)
        assert "MB" in format_bytes(5 * 1024 * 1024)
        assert "GB" in format_bytes(10 * 1024 * 1024 * 1024)


class TestComputeEstimator:
    def test_estimate_training_flops_6nd_heuristic(self):
        n = 7_000_000_000
        d = 100_000_000_000
        res = estimate_training_flops(n, d)
        assert res["total_training_flops"] == 6 * n * d
        assert res["forward_flops"] == 2 * n * d
        assert res["backward_flops"] == 4 * n * d
        assert res["inference_flops_per_token"] == 2 * n

    def test_estimate_chinchilla_optimal_ratios(self):
        budget = 1e20  # FLOPs
        res = estimate_chinchilla_optimal(budget)
        n_opt = res["optimal_parameters"]
        d_opt = res["optimal_tokens"]
        # Ratio D/N should be approximately 20
        ratio = d_opt / n_opt
        assert math.isclose(ratio, 20.0, rel_tol=0.05)
        # Check FLOPs reconstructed
        flops_check = 6 * n_opt * d_opt
        assert math.isclose(flops_check, budget, rel_tol=0.05)

    def test_estimate_chinchilla_negative_budget_raises(self):
        with pytest.raises(ValueError):
            estimate_chinchilla_optimal(-1e18)

    def test_tokens_per_second_calculation(self):
        assert tokens_per_second(10_000, 2.0) == 5000.0
        assert tokens_per_second(0, 5.0) == 0.0
        assert tokens_per_second(100, 0.0) == 0.0

    def test_estimated_training_time_hours_days(self):
        flops = 1e18
        res = estimated_training_time(flops, hardware_tflops=100.0, num_devices=8, model_flops_utilization=0.40)
        assert res["estimated_seconds"] > 0
        assert res["estimated_hours"] == round(res["estimated_seconds"] / 3600.0, 2)
        assert res["estimated_days"] == round(res["estimated_hours"] / 24.0, 4)


class TestScalingLaws:
    def test_power_law_loss_monotonic_decrease(self):
        l1 = power_law_loss(scale=1e6, a=10.0, alpha=0.3, irreducible_loss=1.5)
        l2 = power_law_loss(scale=1e7, a=10.0, alpha=0.3, irreducible_loss=1.5)
        l3 = power_law_loss(scale=1e8, a=10.0, alpha=0.3, irreducible_loss=1.5)
        assert l1 > l2 > l3
        assert l3 > 1.5

    def test_power_law_loss_negative_scale_raises(self):
        with pytest.raises(ValueError):
            power_law_loss(-10, a=5.0, alpha=0.2)

    def test_fit_power_law_recovers_synthetic_curve(self):
        scales = [1e5, 1e6, 1e7, 1e8]
        true_a = 5.0
        true_alpha = 0.25
        e = 1.6
        synthetic_losses = [e + true_a * (s ** (-true_alpha)) for s in scales]

        fit = fit_power_law(scales, synthetic_losses, irreducible_loss=e)
        assert math.isclose(fit["alpha"], true_alpha, rel_tol=1e-2)
        assert math.isclose(fit["A"], true_a, rel_tol=1e-2)

    def test_predict_scaled_loss_function(self):
        pred = predict_scaled_loss(target_scale=1e7, a=4.0, alpha=0.2, irreducible_loss=1.5)
        assert pred > 1.5
        assert isinstance(pred, float)

    def test_analyze_scaling_efficiency_marginal_returns(self):
        sizes = [1e6, 2e6, 4e6]
        losses = [3.0, 2.6, 2.4]
        analysis = analyze_scaling_efficiency(sizes, losses)
        assert len(analysis) == 3
        assert analysis[0]["marginal_improvement"] is None
        assert analysis[1]["marginal_improvement"] == 0.4
        assert analysis[2]["marginal_improvement"] == 0.2
