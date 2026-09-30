"""
Standalone script to calculate training FLOPs (6ND) and Chinchilla compute-optimal ratios.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.compute import estimate_training_flops, estimate_chinchilla_optimal, estimated_training_time


def run_flops_audit():
    training_runs = {
        "MiniGPT Tiny (0.8M, 10k tokens)": (809_088, 10_000),
        "MiniGPT Small (4.7M, 50k tokens)": (4_750_000, 50_000),
        "MiniGPT Medium (25M, 200k tokens)": (25_300_000, 200_000),
        "Chinchilla 7B (7B, 140B tokens)": (7_000_000_000, 140_000_000_000),
        "LLaMA 3 8B (8B, 15T tokens)": (8_000_000_000, 15_000_000_000_000),
        "LLaMA 3 70B (70B, 15T tokens)": (70_000_000_000, 15_000_000_000_000),
    }

    print("=" * 90)
    print(f"{'Run Configuration':<36} | {'Params (N)':<12} | {'Tokens (D)':<14} | {'Training FLOPs (6ND)':<20}")
    print("-" * 90)

    for name, (n, d) in training_runs.items():
        res = estimate_training_flops(n, d)
        flops_sci = f"{res['total_training_flops']:.2e}"
        print(f"{name:<36} | {n:<12.2e} | {d:<14.2e} | {flops_sci:<20}")

    print("=" * 90)
    print("\nCHINCHILLA COMPUTE-OPTIMAL ALLOCATION (C ~= 120 * N^2)")
    print("-" * 90)
    budgets = [1e18, 1e20, 1e22, 1e24]
    for b in budgets:
        opt = estimate_chinchilla_optimal(b)
        n_opt = opt["optimal_parameters"]
        d_opt = opt["optimal_tokens"]
        print(f"  Compute Budget: {b:.1e} FLOPs -> Optimal Params: {n_opt:.2e} | Optimal Tokens: {d_opt:.2e} (Ratio: {opt['tokens_per_parameter_ratio']})")
    print("=" * 90)


if __name__ == "__main__":
    run_flops_audit()
