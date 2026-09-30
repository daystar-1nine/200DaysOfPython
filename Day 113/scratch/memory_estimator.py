"""
Standalone script to calculate model and training memory across precision formats (FP32, FP16, BF16, INT8, INT4).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.params import parameter_memory, estimate_training_memory


def run_memory_audit():
    models = {
        "Tiny (0.8M)": 809_088,
        "Small (4.7M)": 4_750_000,
        "Medium (25M)": 25_300_000,
        "GPT-2 (124M)": 124_000_000,
        "LLaMA (7B)": 7_000_000_000,
        "LLaMA (70B)": 70_000_000_000
    }

    precisions = ["FP32", "FP16", "INT8", "INT4"]

    print("=" * 80)
    print("STATIC PARAMETER MEMORY FOOTPRINT")
    print(f"{'Model':<15} | " + " | ".join(f"{p:<12}" for p in precisions))
    print("-" * 80)

    for name, params in models.items():
        row = [f"{name:<15}"]
        for p in precisions:
            mem = parameter_memory(params, precision=p.lower())
            if mem["gigabytes"] >= 1.0:
                row.append(f"{mem['gigabytes']:>9.2f} GB")
            else:
                row.append(f"{mem['megabytes']:>9.2f} MB")
        print(" | ".join(row))

    print("=" * 80)
    print("\nTRAINING MEMORY BREAKDOWN (Model + Gradients + AdamW Optimizer + Activations)")
    print("-" * 80)
    training_7b = estimate_training_memory(
        parameters=7_000_000_000,
        optimizer="adamw",
        precision="bf16",
        batch_size=4,
        context_length=2048,
        num_layers=32,
        embed_dim=4096,
        num_heads=32
    )
    for k, v in training_7b.items():
        print(f"  {k:<30}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    run_memory_audit()
