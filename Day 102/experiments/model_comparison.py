"""Experiment: Model Comparison across NB, LR, and Linear SVM."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.data.loader import load_data
from app.experiments.model_benchmark import run_benchmark

def run():
    print("=== EXPERIMENT: MODEL BENCHMARK SUITE ===")
    df = load_data(config.DATA_RAW)
    X = df["text"].values
    y = np.where(df["label"] == "spam", 1, 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    bench_df, _ = run_benchmark(X_train, y_train, X_test, y_test)
    print(bench_df[["Model", "Features", "CV Mean", "CV Std", "Accuracy", "F1", "Train Time (ms)", "Inference Time (ms)"]])

if __name__ == "__main__":
    run()
