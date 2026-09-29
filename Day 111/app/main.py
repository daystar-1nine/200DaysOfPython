"""
Main pipeline execution script for Day 111: BERT & Bidirectional Transformers.
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from app.config import (
    RAW_DATA_PATH,
    OUTPUT_DIR,
    CHARTS_DIR,
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    FROZEN_LR,
    FINETUNED_LR
)
from app.data.loader import load_data
from app.data.cleaner import clean_dataset
from app.data.splitter import split_dataset
from app.preprocessing.inputs import build_tensor_dataset, generate_tokenization_examples_csv
from app.training.fine_tuning import run_experiment_a_frozen, run_experiment_b_finetuned
from app.training.benchmarking import run_tfidf_baseline, build_benchmark_table
from app.training.trainer import BertTrainer
from app.evaluation.threshold import sweep_thresholds
from app.evaluation.errors import analyze_errors
from app.visualization.attention import visualize_bert_attention
from app.visualization.embeddings import visualize_cls_embeddings
from app.visualization.benchmark import (
    plot_benchmark_comparison,
    plot_training_curves,
    plot_threshold_curve
)


def run_pipeline() -> None:
    print("=" * 70)
    print("DAY 111: BERT & BIDIRECTIONAL TRANSFORMERS PIPELINE")
    print("=" * 70)

    # 1. Load and clean data
    print("\n[Step 1] Loading and cleaning dataset...")
    df = load_data(RAW_DATA_PATH)
    df_clean = clean_dataset(df)
    train_df, val_df, test_df = split_dataset(df_clean)
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # 2. Tokenizer & Tokenization Demo
    print(f"\n[Step 2] Initializing BERT Tokenizer ({MODEL_NAME})...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    token_csv = OUTPUT_DIR / "tokenization_examples.csv"
    generate_tokenization_examples_csv(tokenizer, token_csv)

    # 3. Create PyTorch Datasets & DataLoaders
    print("\n[Step 3] Building PyTorch TensorDatasets & DataLoaders...")
    train_ds = build_tensor_dataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length=MAX_LENGTH)
    val_ds = build_tensor_dataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length=MAX_LENGTH)
    test_ds = build_tensor_dataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Experiment A: Frozen BERT Backbone
    print("\n[Step 4] Running Experiment A: Frozen BERT Backbone...")
    model_frozen, history_a, summary_a = run_experiment_a_frozen(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=FROZEN_LR,
        model_name=MODEL_NAME
    )

    # 5. Experiment B: Fine-Tuned BERT Backbone
    print("\n[Step 5] Running Experiment B: Fine-Tuned BERT Backbone...")
    model_finetuned, history_b, summary_b = run_experiment_b_finetuned(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=FINETUNED_LR,
        model_name=MODEL_NAME
    )

    # 6. TF-IDF Baseline & Benchmark Table
    print("\n[Step 6] Running TF-IDF Baseline and assembling Benchmark Table...")
    tfidf_res = run_tfidf_baseline(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        test_texts=test_df["text"].tolist(),
        test_labels=test_df["label"].tolist()
    )
    benchmark_df = build_benchmark_table(
        tfidf_metrics=tfidf_res,
        frozen_bert_metrics=summary_a,
        finetuned_bert_metrics=summary_b,
        output_path=OUTPUT_DIR / "benchmark.csv"
    )
    print("\nBenchmark Summary:\n", benchmark_df.to_string(index=False))

    # 7. Threshold Analysis on Fine-Tuned BERT
    print("\n[Step 7] Running Decision Threshold Sweep...")
    finetuned_trainer = BertTrainer(model=model_finetuned)
    eval_results = finetuned_trainer.evaluate(test_loader)
    threshold_df = sweep_thresholds(
        y_true=eval_results["y_true"],
        y_probs=eval_results["y_probs"],
        output_path=OUTPUT_DIR / "threshold_analysis.csv"
    )

    # 8. Error Analysis
    print("\n[Step 8] Running Error Analysis on Test Predictions...")
    errors_df = analyze_errors(
        texts=test_df["text"].tolist(),
        y_true=eval_results["y_true"],
        y_pred=eval_results["y_pred"],
        y_probs=eval_results["y_probs"],
        output_path=OUTPUT_DIR / "errors.csv"
    )

    # 9. Attention Visualization
    print("\n[Step 9] Visualizing Self-Attention Heatmaps...")
    spam_sample = "Congratulations! You won a $1,000 cash prize. Call 555-1234 to claim now!"
    ham_sample = "Hey, are we still meeting for lunch today at noon?"
    visualize_bert_attention(
        text=spam_sample,
        model=model_finetuned,
        tokenizer=tokenizer,
        output_path=CHARTS_DIR / "attention_spam.png",
        title="BERT Self-Attention: Spam Sample"
    )
    visualize_bert_attention(
        text=ham_sample,
        model=model_finetuned,
        tokenizer=tokenizer,
        output_path=CHARTS_DIR / "attention_ham.png",
        title="BERT Self-Attention: Ham Sample"
    )

    # 10. [CLS] Embeddings PCA Visualization
    print("\n[Step 10] Visualizing [CLS] Contextual Embeddings (PCA)...")
    cls_emb, _, labels = finetuned_trainer.extract_representations(test_loader)
    visualize_cls_embeddings(
        cls_embeddings=cls_emb,
        labels=labels,
        output_path=CHARTS_DIR / "cls_embeddings_pca.png",
        title="Fine-Tuned BERT [CLS] Representation Space"
    )

    # 11. Plot Benchmark and Training Curves
    print("\n[Step 11] Generating Benchmark Charts and Training Curves...")
    plot_benchmark_comparison(benchmark_df, CHARTS_DIR / "benchmark_comparison.png")
    plot_training_curves(history_a, history_b, CHARTS_DIR / "training_curves.png")
    plot_threshold_curve(threshold_df, CHARTS_DIR / "threshold_curve.png")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY! All artifacts saved to Day 111/output/")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
