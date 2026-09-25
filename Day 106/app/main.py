"""
Main orchestration pipeline for Day 106: RNNs & Sequential Text Learning.
Executes data preprocessing, baseline evaluation, SimpleRNN training, hyperparameter sweeps,
threshold analysis, error analysis, visualization generation, and report compilation.
"""

from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch

# Ensure Day 106 is in path
CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from app.config import (
    RAW_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    VOCAB_PATH, TOKENIZER_PATH, MODEL_CHECKPOINT_PATH,
    OUTPUT_DIR, CHARTS_DIR, EMBEDDING_DIM, HIDDEN_UNITS, DENSE_UNITS,
    MAX_SEQ_LENGTH, DROPOUT_1, DROPOUT_2, LEARNING_RATE, BATCH_SIZE,
    EPOCHS, PATIENCE, RANDOM_SEED, DEFAULT_THRESHOLD
)
from app.data import DataLoader, DataCleaner, DataValidator, DataSplitter, EnvironmentAuditor
from app.preprocessing import WordTokenizer, Vocabulary, TextEncoder, SequencePadder
from app.models import SimpleRNNClassifier, BaselineModels, Day105PoolingClassifier
from app.training import RNNTrainer, EarlyStopping, ModelCheckpoint, ExperimentRunner
from app.evaluation import (
    ClassificationMetrics, ConfusionMatrixCalculator,
    ROCEvaluator, PrecisionRecallEvaluator, ThresholdAnalyzer
)
from app.analysis import ErrorAnalyzer, EmbeddingAnalyzer, ModelInsights
from app.visualization import (
    plot_dataset_distributions, plot_training_curves,
    plot_evaluation_charts, plot_experiment_and_comparison_charts
)
from app.report import ReportGenerator


def main():
    # Set stdout encoding for clean console output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 70)
    print("[DAY 106] RNNs & SEQUENTIAL TEXT LEARNING ENGINE")
    print("=" * 70 + "\n")

    # Set reproducibility seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Audit Runtime Environment
    print("[1/10] Auditing Environment Runtime...")
    audit_info = EnvironmentAuditor.audit()
    print(f"  • Python Version : {audit_info['python_version']} ({audit_info['platform']})")
    print(f"  • PyTorch        : {audit_info['pytorch_status']}")
    print(f"  • TensorFlow     : {audit_info['tensorflow_status']}")

    # 2. Ingest, Clean & Validate Dataset
    print("\n[2/10] Loading & Validating Dataset...")
    df_raw = DataLoader.load_csv(RAW_DATA_PATH)
    df_clean = DataCleaner.clean_dataframe(df_raw)
    is_valid, errors = DataValidator.validate(df_clean)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {errors}")
    ham_count = (df_clean["label"] == "ham").sum()
    spam_count = (df_clean["label"] == "spam").sum()
    print(f"[OK] Validated {len(df_clean)} samples. Ham: {ham_count}, Spam: {spam_count}")

    # 3. Stratified Partitioning (70/15/15)
    print("\n[3/10] Performing Stratified Train/Val/Test Split (70/15/15)...")
    train_df, val_df, test_df = DataSplitter.split_and_save(
        df_clean, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=RANDOM_SEED
    )
    print(f"  • Train: {len(train_df)} samples | Val: {len(val_df)} samples | Test: {len(test_df)} samples")

    # 4. Vocabulary Construction (Train Only - Zero Leakage)
    print("\n[4/10] Building Vocabulary strictly on Training Data...")
    vocabulary = Vocabulary(min_freq=2).fit(train_df["text"])
    vocabulary.save(VOCAB_PATH)
    vocabulary.save(TOKENIZER_PATH)
    print(f"[OK] Vocabulary constructed with {len(vocabulary)} tokens (<PAD>=0, <UNK>=1).")

    # 5. Preprocessing & Encoding
    print(f"\n[5/10] Encoding & Padding Sequences (T = {MAX_SEQ_LENGTH})...")
    encoder = TextEncoder(vocabulary)
    train_ids = encoder.encode_batch(train_df["text"].tolist())
    val_ids = encoder.encode_batch(val_df["text"].tolist())
    test_ids = encoder.encode_batch(test_df["text"].tolist())

    X_train, M_train = SequencePadder.pad_batch(train_ids, MAX_SEQ_LENGTH)
    X_val, M_val = SequencePadder.pad_batch(val_ids, MAX_SEQ_LENGTH)
    X_test, M_test = SequencePadder.pad_batch(test_ids, MAX_SEQ_LENGTH)

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    train_loader = RNNTrainer.create_dataloader(X_train, M_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = RNNTrainer.create_dataloader(X_val, M_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = RNNTrainer.create_dataloader(X_test, M_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Benchmark Classical & Day 105 Baselines
    print("\n[6/10] Training & Evaluating Baselines (Dummy, TF-IDF LogReg, TF-IDF SVM, Day 105 Pooling)...")
    benchmark_records = []

    # 6a. Dummy Classifier
    dummy = BaselineModels.build_dummy_classifier()
    dummy.fit(train_df["text"], y_train)
    dummy_preds = dummy.predict(test_df["text"])
    dummy_probs = np.zeros(len(test_df))
    dummy_metrics = ClassificationMetrics.compute(y_test, dummy_probs)
    benchmark_records.append({
        "model": "Dummy Classifier",
        "representation": "Majority Class",
        "f1": dummy_metrics["f1"],
        "accuracy": dummy_metrics["accuracy"],
        "precision": dummy_metrics["precision"],
        "recall": dummy_metrics["recall"],
        "roc_auc": dummy_metrics["roc_auc"],
        "parameters": 0,
        "training_time": 0.001
    })

    # 6b. TF-IDF + Logistic Regression
    tfidf_lr = BaselineModels.build_tfidf_logreg(RANDOM_SEED)
    tfidf_lr.fit(train_df["text"].tolist(), y_train)
    lr_metrics = tfidf_lr.evaluate(test_df["text"].tolist(), y_test)
    benchmark_records.append({
        "model": "Logistic Regression",
        "representation": "TF-IDF (1-2 N-grams)",
        "f1": lr_metrics["f1"],
        "accuracy": lr_metrics["accuracy"],
        "precision": lr_metrics["precision"],
        "recall": lr_metrics["recall"],
        "roc_auc": lr_metrics["roc_auc"],
        "parameters": lr_metrics["num_features"],
        "training_time": lr_metrics["training_time"]
    })

    # 6c. TF-IDF + Linear SVM
    tfidf_svm = BaselineModels.build_tfidf_svm(RANDOM_SEED)
    tfidf_svm.fit(train_df["text"].tolist(), y_train)
    svm_metrics = tfidf_svm.evaluate(test_df["text"].tolist(), y_test)
    benchmark_records.append({
        "model": "Linear SVM",
        "representation": "TF-IDF (1-2 N-grams)",
        "f1": svm_metrics["f1"],
        "accuracy": svm_metrics["accuracy"],
        "precision": svm_metrics["precision"],
        "recall": svm_metrics["recall"],
        "roc_auc": svm_metrics["roc_auc"],
        "parameters": svm_metrics["num_features"],
        "training_time": svm_metrics["training_time"]
    })

    # 6d. Day 105 Neural Pooling Classifier
    pool_model = Day105PoolingClassifier(len(vocabulary), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_UNITS)
    pool_trainer = RNNTrainer(pool_model, learning_rate=0.001)
    pool_start = time.perf_counter()
    pool_trainer.fit(train_loader, val_loader, epochs=15, verbose=False)
    pool_time = time.perf_counter() - pool_start
    pool_probs = torch.sigmoid(pool_model(torch.tensor(X_test, dtype=torch.long), torch.tensor(M_test, dtype=torch.float32))).detach().numpy()
    pool_metrics = ClassificationMetrics.compute(y_test, pool_probs)
    benchmark_records.append({
        "model": "Day 105 Neural Model",
        "representation": "Embedding + Masked Pooling",
        "f1": pool_metrics["f1"],
        "accuracy": pool_metrics["accuracy"],
        "precision": pool_metrics["precision"],
        "recall": pool_metrics["recall"],
        "roc_auc": pool_metrics["roc_auc"],
        "parameters": pool_model.count_parameters(),
        "training_time": pool_time
    })

    # 7. Train Day 106 SimpleRNN Classifier
    print("\n[7/10] Training SimpleRNN Text Classifier (Embedding 64 -> RNN 64 -> Dense 32 -> Output)...")
    rnn_model = SimpleRNNClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_units=HIDDEN_UNITS,
        dense_units=DENSE_UNITS,
        dropout_1=DROPOUT_1,
        dropout_2=DROPOUT_2
    )
    param_counts = rnn_model.count_parameters()
    print(f"  • Model Parameters: Total = {param_counts['total']:,} "
          f"(Embedding: {param_counts['embedding']:,}, RNN: {param_counts['rnn']:,}, Head: {param_counts['head']:,})")

    early_stopping = EarlyStopping(patience=PATIENCE)
    checkpoint = ModelCheckpoint(filepath=MODEL_CHECKPOINT_PATH)
    trainer = RNNTrainer(rnn_model, learning_rate=LEARNING_RATE, grad_clip=5.0)

    start_train = time.perf_counter()
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        verbose=True
    )
    rnn_train_time = time.perf_counter() - start_train

    # 8. Evaluate SimpleRNN on Test Data
    print("\n[8/10] Evaluating SimpleRNN on Held-Out Test Set...")
    t_ids_test = torch.tensor(X_test, dtype=torch.long)
    t_mask_test = torch.tensor(M_test, dtype=torch.float32)
    test_probs = rnn_model.predict_proba(t_ids_test, t_mask_test)
    test_preds = (test_probs >= DEFAULT_THRESHOLD).astype(int)

    rnn_metrics = ClassificationMetrics.compute(y_test, test_probs, threshold=DEFAULT_THRESHOLD)
    benchmark_records.append({
        "model": "Day 106 SimpleRNN",
        "representation": "Embedding + SimpleRNN (H=64)",
        "f1": rnn_metrics["f1"],
        "accuracy": rnn_metrics["accuracy"],
        "precision": rnn_metrics["precision"],
        "recall": rnn_metrics["recall"],
        "roc_auc": rnn_metrics["roc_auc"],
        "parameters": param_counts["total"],
        "training_time": rnn_train_time
    })

    benchmark_df = pd.DataFrame(benchmark_records)
    print("\n" + "=" * 90)
    print(f"{'Model':<25} | {'Representation':<28} | {'F1':<8} | {'Accuracy':<8} | {'Params':<8} | {'Time (s)':<8}")
    print("-" * 90)
    for _, r in benchmark_df.iterrows():
        print(f"{r['model']:<25} | {r['representation']:<28} | {r['f1']:<8.4f} | {r['accuracy']:<8.4f} | {r['parameters']:<8,d} | {r['training_time']:<8.3f}")
    print("=" * 90 + "\n")

    # Save Predictions & Benchmark Metrics
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame({
        "text": test_df["text"],
        "actual_label": test_df["label"],
        "actual_target": y_test,
        "predicted_label": ["spam" if p == 1 else "ham" for p in test_preds],
        "predicted_target": test_preds,
        "probability": np.round(test_probs, 4)
    })
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    benchmark_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    # 9. Threshold Analysis (0.10 to 0.90)
    print("[9/10] Performing Decision Threshold Sweeps (0.10 to 0.90)...")
    threshold_df = ThresholdAnalyzer.evaluate_thresholds(y_test, test_probs)
    threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
    print("  • Saved threshold analysis to output/threshold_analysis.csv")

    # Error Analysis
    error_df = ErrorAnalyzer.analyze(test_df["text"].tolist(), y_test, test_probs, threshold=DEFAULT_THRESHOLD)
    error_df.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)
    print(f"  • Diagnosed {len(error_df)} misclassified samples (saved to output/error_analysis.csv)")

    # 10. Run Controlled Experiments A, B, C, D
    print("\n[10/10] Running Controlled Experiments (A: 32/32, B: 64/64, C: 128/128, D: T in [20, 40, 60, 100])...")
    runner = ExperimentRunner(train_df, val_df, test_df, vocabulary, random_seed=RANDOM_SEED)
    experiments_df = runner.run_all(epochs=15)
    experiments_df.to_csv(OUTPUT_DIR / "experiments.csv", index=False)
    print("  • Experimental sweep results saved to output/experiments.csv")

    # Extract Embeddings and PCA
    weights = EmbeddingAnalyzer.extract_weights(rnn_model)
    emb_df = EmbeddingAnalyzer.project_pca(weights, vocabulary)
    EmbeddingAnalyzer.save_embeddings(emb_df, OUTPUT_DIR / "embeddings.csv")
    print("  • 2D PCA projected word embeddings saved to output/embeddings.csv")

    # Generate Visualizations (18 Charts)
    print("\nGenerating all 18 Visualizations...")
    cm_dict = ConfusionMatrixCalculator.compute(y_test, test_preds)
    roc_dict = ROCEvaluator.compute_curve(y_test, test_probs)
    pr_dict = PrecisionRecallEvaluator.compute_curve(y_test, test_probs)

    plot_dataset_distributions(df_clean, CHARTS_DIR)
    plot_training_curves(history, CHARTS_DIR)
    plot_evaluation_charts(cm_dict, roc_dict, pr_dict, threshold_df, CHARTS_DIR)
    plot_experiment_and_comparison_charts(experiments_df, benchmark_df, CHARTS_DIR)
    print(f"[OK] Generated 18 publication-quality charts in {CHARTS_DIR}")

    # Generate Markdown Reports
    charts_list = [f"{i}.png" for i in range(1, 19)]
    report_md = ReportGenerator.generate(
        audit_info=audit_info,
        benchmark_df=benchmark_df,
        experiments_df=experiments_df,
        threshold_df=threshold_df,
        error_df=error_df,
        charts_list=charts_list,
        output_path=OUTPUT_DIR / "report.md"
    )
    shutil.copy(OUTPUT_DIR / "report.md", CURRENT_DIR / "DAY_106_REPORT.md")
    print(f"[OK] Generated full engineering report: DAY_106_REPORT.md")
    print("\n[SUCCESS] Day 106 Neural NLP & Sequential Text Learning Pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
