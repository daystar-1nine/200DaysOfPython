"""
Compute and FLOPs estimation module for LLM training and inference.
Implements the 6ND training FLOPs heuristic and Chinchilla compute-optimal trade-offs.
"""
import math
from typing import Dict, Any


def estimate_training_flops(parameters: int, tokens: int) -> Dict[str, Any]:
    """
    Estimates total Floating Point Operations (FLOPs) required to train a Transformer model.
    Standard heuristic (Kaplan et al., Chinchilla):
      Forward pass: ~ 2 * N FLOPs per token
      Backward pass: ~ 4 * N FLOPs per token (activation recomputation or gradient accumulation)
      Total per token: ~ 6 * N FLOPs
      Total training compute: ~ 6 * N * D FLOPs
    """
    forward_flops = 2 * parameters * tokens
    backward_flops = 4 * parameters * tokens
    total_training_flops = 6 * parameters * tokens
    inference_flops_per_token = 2 * parameters

    return {
        "parameters": parameters,
        "tokens": tokens,
        "forward_flops": forward_flops,
        "backward_flops": backward_flops,
        "total_training_flops": total_training_flops,
        "inference_flops_per_token": inference_flops_per_token,
        "petaflops_days": round(total_training_flops / (1e15 * 86400), 4)
    }


def estimate_chinchilla_optimal(compute_budget_flops: float) -> Dict[str, Any]:
    """
    Computes compute-optimal model size (parameters N) and training dataset size (tokens D)
    for a given compute budget C under Chinchilla scaling laws (Hoffmann et al., 2022).
    
    C ≈ 6 * N * D
    In Chinchilla, optimal parameter scaling is balanced 1:1 with token scaling:
      D / N ≈ 20 tokens per parameter
      C ≈ 6 * N * (20 * N) = 120 * N^2
      N_opt = sqrt(C / 120)
      D_opt = 20 * N_opt
    """
    if compute_budget_flops <= 0:
        raise ValueError("Compute budget must be strictly positive.")

    n_optimal = math.sqrt(compute_budget_flops / 120.0)
    d_optimal = 20.0 * n_optimal

    return {
        "compute_budget_flops": compute_budget_flops,
        "optimal_parameters": int(round(n_optimal)),
        "optimal_tokens": int(round(d_optimal)),
        "tokens_per_parameter_ratio": round(d_optimal / n_optimal, 2),
        "check_flops": 6 * int(round(n_optimal)) * int(round(d_optimal))
    }


def tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float:
    """Computes training throughput in tokens per second (TPS)."""
    if elapsed_seconds <= 0:
        return 0.0
    return round(num_tokens / elapsed_seconds, 2)


def estimated_training_time(
    flops: float,
    hardware_tflops: float,
    num_devices: int = 1,
    model_flops_utilization: float = 0.35
) -> Dict[str, Any]:
    """
    Estimates real-world wall-clock training time given:
    - Total FLOPs
    - Peak theoretical TFLOPs per accelerator (e.g. 312 TFLOPs for A100 FP16)
    - Number of accelerators
    - Model FLOPs Utilization (MFU) (typically 0.30 - 0.45 in production)
    """
    if hardware_tflops <= 0 or num_devices <= 0 or model_flops_utilization <= 0:
        raise ValueError("Hardware specifications and MFU must be positive.")

    peak_flops_per_sec = hardware_tflops * 1e12 * num_devices
    achieved_flops_per_sec = peak_flops_per_sec * model_flops_utilization

    total_seconds = flops / achieved_flops_per_sec
    total_hours = total_seconds / 3600.0
    total_days = total_hours / 24.0

    return {
        "total_flops": flops,
        "hardware_peak_tflops_total": hardware_tflops * num_devices,
        "mfu": model_flops_utilization,
        "achieved_tflops_total": round((achieved_flops_per_sec / 1e12), 2),
        "estimated_seconds": round(total_seconds, 2),
        "estimated_hours": round(total_hours, 2),
        "estimated_days": round(total_days, 4)
    }
