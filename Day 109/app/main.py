import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.training.benchmark import run_attention_benchmark

def main():
    print("🚀 Starting Day 109: Attention Mechanism & Text Classification Pipeline")
    results = run_attention_benchmark()
    print("\nFinal Model Comparison Benchmark:")
    print(results["benchmark_df"][["model", "parameters", "training_time_sec", "f1", "roc_auc"]])
    print("\n✅ Day 109 Attention Execution Finished!")

if __name__ == "__main__":
    main()
