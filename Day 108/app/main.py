import sys
from pathlib import Path

# Add project root to sys.path so app is cleanly importable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.training.benchmark import run_controlled_benchmark

def main():
    print("🚀 Starting Day 108: GRU & Recurrent Model Benchmarking Pipeline")
    results = run_controlled_benchmark()
    print("\nBenchmark Summary:")
    print(results["comparison_df"][["model", "parameters", "training_time_sec", "f1", "roc_auc"]])
    print("\n✅ Day 108 Benchmark Execution Succeeded!")

if __name__ == "__main__":
    main()
