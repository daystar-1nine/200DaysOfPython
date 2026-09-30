"""
Standalone script to calculate parameter counts across Tiny, Small, Medium, and 7B LLM architectures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.params import estimate_parameters


def run_parameter_audit():
    configs = {
        "Tiny (Day 113)": {"vocab": 64, "d": 128, "layers": 4, "heads": 4, "ctx": 64},
        "Small (Day 113)": {"vocab": 64, "d": 256, "layers": 6, "heads": 8, "ctx": 64},
        "Medium (Day 113)": {"vocab": 64, "d": 512, "layers": 8, "heads": 8, "ctx": 64},
        "GPT-2 Small (124M)": {"vocab": 50257, "d": 768, "layers": 12, "heads": 12, "ctx": 1024},
        "LLaMA-style 7B": {"vocab": 32000, "d": 4096, "layers": 32, "heads": 32, "ctx": 4096},
    }

    print("=" * 80)
    print(f"{'Model Architecture':<22} | {'Params (M)':<12} | {'Blocks (M)':<12} | {'Embeddings (M)':<15}")
    print("-" * 80)

    for name, c in configs.items():
        res = estimate_parameters(
            vocab_size=c["vocab"],
            embed_dim=c["d"],
            num_layers=c["layers"],
            num_heads=c["heads"],
            context_length=c["ctx"],
            tie_weights=True
        )
        total_m = res["total_parameters"] / 1e6
        blocks_m = res["all_blocks"] / 1e6
        embeds_m = (res["token_embeddings"] + res["position_embeddings"]) / 1e6
        print(f"{name:<22} | {total_m:>10.2f} M | {blocks_m:>10.2f} M | {embeds_m:>13.2f} M")

    print("=" * 80)


if __name__ == "__main__":
    run_parameter_audit()
