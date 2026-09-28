"""
Main execution pipeline for Day 110: Transformers From Scratch.
Orchestrates data preprocessing, model training, benchmarking, ablation study,
experimental sweeps, attention visualization, and metric reporting.
"""
from pathlib import Path
import time
import pandas as pd
import numpy as np
import torch

from app.config import config
from app.data.loader import load_raw_sms_data
from app.data.cleaner import clean_sms_data
from app.data.validator import validate_sms_data
from app.data.splitter import split_dataset
from app.preprocessing.tokenizer import tokenize_text
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.padding import pad_sequences
from app.preprocessing.positional_encoding import get_positional_encoding_numpy
from app.models.classifier import MiniTransformerClassifier
from app.models.baselines import RecurrentBaselineClassifier
from app.training.trainer import Trainer
from app.training.benchmark import run_comprehensive_benchmark
from app.evaluation.metrics import compute_metrics
from app.evaluation.confusion import compute_confusion_matrix, plot_confusion_matrix
from app.evaluation.threshold import analyze_thresholds, plot_threshold_curves
from app.evaluation.roc import plot_roc_pr_curves
from app.analysis.ablation import run_ablation_study
from app.analysis.attention_analysis import extract_sample_attention, compute_head_entropy
from app.analysis.error_analysis import perform_error_analysis
from app.visualization.positional_encoding import save_positional_encoding_heatmap
from app.visualization.attention_heatmap import plot_multi_head_attention_grid, plot_single_head_attention
from app.visualization.benchmark_plots import (
    plot_training_history,
    plot_benchmark_comparison,
    plot_parameters_vs_time,
    plot_ablation_comparison,
    plot_sequence_length_complexity,
    plot_experiment_sweep
)


import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    print("=" * 70)
    print("[DAY 110] TRANSFORMERS FROM SCRATCH - EXECUTION PIPELINE")
    print("=" * 70)

    # 1. Load and Clean Dataset
    print("\n[1/8] Loading and preprocessing dataset...")
    df_raw = load_raw_sms_data(config.RAW_DATA_FILE)
    df_clean = clean_sms_data(df_raw)
    stats = validate_sms_data(df_clean)
    print(f"Total samples: {stats['total_samples']} (Spam: {stats['spam_count']}, Ham: {stats['ham_count']})")

    # 2. Stratified 3-way Split
    print("\n[2/8] Partitioning dataset into train/val/test splits...")
    train_df, val_df, test_df = split_dataset(
        df_clean,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Tokenization & Vocabulary (Fitted strictly on Train!)
    print("\n[3/8] Building vocabulary (Train-split strict zero-leakage)...")
    train_tokens = [tokenize_text(t) for t in train_df["text"]]
    val_tokens = [tokenize_text(t) for t in val_df["text"]]
    test_tokens = [tokenize_text(t) for t in test_df["text"]]

    vocab = Vocabulary(min_freq=config.MIN_FREQ)
    vocab.fit(train_tokens)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size} tokens")

    # 4. Encoding and Padding
    x_train = pad_sequences([vocab.encode(t) for t in train_tokens], max_length=config.MAX_LENGTH)
    x_val = pad_sequences([vocab.encode(t) for t in val_tokens], max_length=config.MAX_LENGTH)
    x_test = pad_sequences([vocab.encode(t) for t in test_tokens], max_length=config.MAX_LENGTH)

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # 5. Sinusoidal Positional Encoding Visualization
    print("\n[4/8] Generating Sinusoidal Positional Encoding visualization...")
    pe_numpy = get_positional_encoding_numpy(config.MAX_LENGTH, config.D_MODEL)
    save_positional_encoding_heatmap(pe_numpy, config.CHARTS_DIR / "positional_encoding.png")
    print(f"Saved: {config.CHARTS_DIR / 'positional_encoding.png'}")

    # 6. Comprehensive Architecture Benchmark
    print("\n[5/8] Running Comprehensive Benchmark across 6 architectures...")
    benchmark_df, test_probs_dict = run_comprehensive_benchmark(
        vocab_size=vocab_size,
        max_length=config.MAX_LENGTH,
        texts_train=train_df["text"].tolist(),
        texts_test=test_df["text"].tolist(),
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        epochs=config.EPOCHS,
        output_csv=config.OUTPUT_DIR / "benchmark.csv"
    )
    print("\nBenchmark Results:")
    print(benchmark_df.to_string(index=False))

    plot_benchmark_comparison(benchmark_df, save_path=config.CHARTS_DIR / "benchmark_comparison.png")
    plot_parameters_vs_time(benchmark_df, save_path=config.CHARTS_DIR / "parameters_vs_time.png")
    plot_roc_pr_curves(y_test, test_probs_dict, save_path=config.CHARTS_DIR / "roc_pr_curves.png")

    # 7. Primary Mini Transformer Model Training & Threshold Analysis
    print("\n[6/8] Training Primary Mini Transformer Classifier...")
    primary_model = MiniTransformerClassifier(
        vocab_size=vocab_size,
        max_length=config.MAX_LENGTH,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    primary_trainer = Trainer(primary_model, learning_rate=config.LEARNING_RATE)
    history, train_time = primary_trainer.fit(
        x_train, y_train, x_val, y_val,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        patience=config.PATIENCE,
        checkpoint_path=config.MODELS_DIR / "transformer_checkpoint.pt"
    )

    plot_training_history(history, save_path=config.CHARTS_DIR / "transformer_training_history.png")

    test_probs = primary_trainer.predict_proba(x_test)
    thresh_df, best_thresh, best_metrics = analyze_thresholds(y_test, test_probs)
    thresh_df.to_csv(config.OUTPUT_DIR / "threshold_analysis.csv", index=False)
    plot_threshold_curves(thresh_df, best_thresh, save_path=config.CHARTS_DIR / "threshold_tradeoff.png")

    cm, cm_counts = compute_confusion_matrix(y_test, test_probs, threshold=best_thresh)
    plot_confusion_matrix(cm, title="Mini Transformer Confusion Matrix", save_path=config.CHARTS_DIR / "transformer_confusion_matrix.png")

    metrics_df = pd.DataFrame([best_metrics])
    metrics_df.to_csv(config.OUTPUT_DIR / "metrics.csv", index=False)

    # 8. Ablation Study
    print("\n[7/8] Running Transformer Ablation Study (Positional, Residual, LayerNorm, Mask)...")
    ablation_df = run_ablation_study(
        vocab_size=vocab_size,
        max_length=config.MAX_LENGTH,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        epochs=config.EPOCHS,
        output_csv=config.OUTPUT_DIR / "ablation.csv"
    )
    print("\nAblation Results:")
    print(ablation_df.to_string(index=False))
    plot_ablation_comparison(ablation_df, save_path=config.CHARTS_DIR / "ablation_study.png")

    # 9. Experiments 1 - 4
    print("\n[8/8] Executing 4 Controlled Experiments...")

    # Experiment 1: Number of Heads
    print(" - Experiment 1: Number of Heads (2, 4, 8)")
    exp1_f1, exp1_times = [], []
    for h in config.EXPERIMENT_HEADS:
        m = MiniTransformerClassifier(vocab_size=vocab_size, max_length=config.MAX_LENGTH, d_model=config.D_MODEL, num_heads=h)
        tr = Trainer(m, learning_rate=config.LEARNING_RATE)
        _, t = tr.fit(x_train, y_train, x_val, y_val, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, patience=config.PATIENCE)
        p = tr.predict_proba(x_test)
        exp1_f1.append(compute_metrics(y_test, p)["f1"])
        exp1_times.append(t)
    plot_experiment_sweep("Number of Attention Heads", config.EXPERIMENT_HEADS, exp1_f1, exp1_times, save_path=config.CHARTS_DIR / "exp1_num_heads.png")

    # Experiment 2: Number of Layers
    print(" - Experiment 2: Number of Layers (1, 2, 4)")
    exp2_f1, exp2_times = [], []
    for l in config.EXPERIMENT_LAYERS:
        m = MiniTransformerClassifier(vocab_size=vocab_size, max_length=config.MAX_LENGTH, d_model=config.D_MODEL, num_heads=config.NUM_HEADS, num_layers=l)
        tr = Trainer(m, learning_rate=config.LEARNING_RATE)
        _, t = tr.fit(x_train, y_train, x_val, y_val, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, patience=config.PATIENCE)
        p = tr.predict_proba(x_test)
        exp2_f1.append(compute_metrics(y_test, p)["f1"])
        exp2_times.append(t)
    plot_experiment_sweep("Number of Transformer Layers", config.EXPERIMENT_LAYERS, exp2_f1, exp2_times, save_path=config.CHARTS_DIR / "exp2_num_layers.png")

    # Experiment 3: Model Dimension
    print(" - Experiment 3: Model Dimension (64, 128, 256)")
    exp3_f1, exp3_times = [], []
    for d in config.EXPERIMENT_DIMS:
        heads = 4 if d % 4 == 0 else 2
        m = MiniTransformerClassifier(vocab_size=vocab_size, max_length=config.MAX_LENGTH, d_model=d, num_heads=heads, d_ff=d*2)
        tr = Trainer(m, learning_rate=config.LEARNING_RATE)
        _, t = tr.fit(x_train, y_train, x_val, y_val, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, patience=config.PATIENCE)
        p = tr.predict_proba(x_test)
        exp3_f1.append(compute_metrics(y_test, p)["f1"])
        exp3_times.append(t)
    plot_experiment_sweep("Model Dimension (d_model)", config.EXPERIMENT_DIMS, exp3_f1, exp3_times, save_path=config.CHARTS_DIR / "exp3_model_dim.png")

    # Experiment 4: Sequence Length Complexity (Transformer vs GRU)
    print(" - Experiment 4: Sequence Length Complexity (32, 64, 100, 150)")
    seq_trans_times, seq_gru_times = [], []
    for seq_len in config.EXPERIMENT_SEQS:
        # Repad sequences to current length
        xt = pad_sequences([vocab.encode(t) for t in train_tokens], max_length=seq_len)
        xv = pad_sequences([vocab.encode(t) for t in val_tokens], max_length=seq_len)

        # Transformer
        m_tf = MiniTransformerClassifier(vocab_size=vocab_size, max_length=seq_len, d_model=64, num_heads=4, d_ff=128, num_layers=1)
        tr_tf = Trainer(m_tf, learning_rate=config.LEARNING_RATE)
        _, t_tf = tr_tf.fit(xt, y_train, xv, y_val, epochs=5, batch_size=32, patience=5)
        seq_trans_times.append(t_tf)

        # GRU
        m_gru = RecurrentBaselineClassifier("gru", vocab_size=vocab_size, embedding_dim=64, hidden_dim=64)
        tr_gru = Trainer(m_gru, learning_rate=config.LEARNING_RATE)
        _, t_gru = tr_gru.fit(xt, y_train, xv, y_val, epochs=5, batch_size=32, patience=5)
        seq_gru_times.append(t_gru)

    plot_sequence_length_complexity(config.EXPERIMENT_SEQS, seq_trans_times, seq_gru_times, save_path=config.CHARTS_DIR / "exp4_seq_length_complexity.png")

    # Attention Maps & Visualization for representative messages
    print("\nVisualizing Multi-Head Attention Maps for Representative Samples...")
    for idx, sample_type in [(0, "sample_01"), (1, "sample_02"), (2, "sample_03")]:
        sample_tokens = test_tokens[idx][:12]  # limit to first 12 tokens for legible heatmaps
        if not sample_tokens:
            continue
        sample_ids = pad_sequences([vocab.encode(sample_tokens)], max_length=config.MAX_LENGTH)[0]
        layer_weights = extract_sample_attention(primary_model, sample_ids)

        for l_idx, lw in enumerate(layer_weights):
            # Save layer grid
            grid_path = config.CHARTS_DIR / f"{sample_type}_layer_{l_idx+1}_grid.png"
            plot_multi_head_attention_grid(sample_tokens, lw, layer_idx=l_idx+1, save_path=grid_path)

            # Save individual heads in transformer_attention
            for h_idx in range(lw.shape[0]):
                head_path = config.ATTENTION_DIR / f"{sample_type}_layer_{l_idx+1}_head_{h_idx+1}.png"
                plot_single_head_attention(sample_tokens, lw[h_idx], title=f"{sample_type.upper()} Layer {l_idx+1} Head {h_idx+1}", save_path=head_path)

    # Error Analysis
    print("Performing error analysis on test set...")
    err_df = perform_error_analysis(test_df["text"].tolist(), y_test, test_probs, threshold=best_thresh)
    err_df.to_csv(config.OUTPUT_DIR / "error_analysis.csv", index=False)

    print("\n" + "=" * 70)
    print("[SUCCESS] DAY 110 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Output files saved to: {config.OUTPUT_DIR}")
    print(f"Charts saved to: {config.CHARTS_DIR}")
    print(f"Attention maps saved to: {config.ATTENTION_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
