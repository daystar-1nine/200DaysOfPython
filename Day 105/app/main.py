"""
Master pipeline execution for Day 105: Neural NLP & Text Classification.
Orchestrates data preparation, model training, evaluation, error diagnosis,
and visualization generation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Safe stdout for Windows cp1252 consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add app parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import config
from app.data.loader import DataLoader
from app.data.validator import DataValidator
from app.data.splitter import DataSplitter
from app.data.environment_check import EnvironmentChecker
from app.preprocessing.tokenizer import Tokenizer
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.encoder import TextEncoder
from app.preprocessing.padding import pad_batch
from app.embeddings.visualization import EmbeddingVisualizer
from app.models.baseline import TfidfBaseline
from app.models.architectures import build_model_c
from app.training.callbacks import EarlyStopping
from app.training.checkpointing import ModelCheckpoint
from app.training.trainer import Trainer
from app.evaluation.metrics import compute_classification_metrics
from app.evaluation.confusion_matrix import compute_confusion_matrix, format_confusion_matrix
from app.evaluation.error_analysis import ErrorAnalyzer
from app.visualization.charts import Visualizer
from app.report import ReportGenerator


def run_pipeline():
    print("=" * 70)
    print("[DAY 105] NEURAL NLP & TEXT CLASSIFICATION ENGINE")
    print("=" * 70)

    config.ensure_directories()

    # 1. Environment Verification Audit
    print("\n[1/8] Auditing Runtime Environment...")
    env_audit = EnvironmentChecker.audit()
    print(f"  • Python Version: {env_audit['python_version'].split()[0]}")
    print(f"  • TensorFlow: {env_audit['tensorflow_status']} ({env_audit['tensorflow_message']})")
    print(f"  • PyTorch: {env_audit['pytorch_status']} ({env_audit['pytorch_message']})")

    # 2. Load & Validate Raw Data
    print("\n[2/8] Loading & Validating SMS Dataset...")
    raw_df = DataLoader.load_raw_data(config.RAW_DATA_PATH)
    encoded_df = DataLoader.encode_labels(raw_df)

    is_valid, errors = DataValidator.validate(encoded_df)
    if not is_valid:
        print(f"[ERROR] Data validation failed: {errors}")
        sys.exit(1)

    dataset_summary = DataValidator.get_summary(encoded_df)
    print(f"[OK] Validated {dataset_summary['total_samples']} samples. Ham: {dataset_summary['class_counts']['ham']}, Spam: {dataset_summary['class_counts']['spam']}")

    # 3. Stratified Partitioning (No Data Leakage)
    print("\n[3/8] Performing Stratified Train/Val/Test Split (70/15/15)...")
    train_df, val_df, test_df = DataSplitter.split(
        encoded_df,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )
    DataSplitter.save_splits(train_df, val_df, test_df, config.PROCESSED_DATA_DIR)
    print(f"  • Train: {len(train_df)} samples | Val: {len(val_df)} samples | Test: {len(test_df)} samples")

    # 4. Tokenization & Train-Only Vocabulary
    print("\n[4/8] Building Vocabulary strictly on Training Data...")
    tokenizer = Tokenizer()
    train_tokens = [tokenizer.tokenize(t) for t in train_df["text"]]

    vocab = Vocabulary(pad_id=config.PAD_ID, unk_id=config.UNK_ID)
    vocab.fit(train_tokens, min_freq=config.MIN_WORD_FREQ)
    vocab_path = config.PROCESSED_DATA_DIR / "vocabulary.json"
    vocab.save(vocab_path)
    print(f"[OK] Vocabulary constructed with {vocab.vocab_size} tokens (including <PAD>=0, <UNK>=1).")

    # 5. Integer Sequence Encoding & Padding
    print("\n[5/8] Encoding & Padding Sequences (T = 40)...")
    encoder = TextEncoder(tokenizer, vocab)

    train_seqs = encoder.encode_corpus(train_df["text"].tolist())
    val_seqs = encoder.encode_corpus(val_df["text"].tolist())
    test_seqs = encoder.encode_corpus(test_df["text"].tolist())

    X_train, M_train = pad_batch(train_seqs, max_length=config.MAX_SEQ_LEN, pad_id=config.PAD_ID)
    X_val, M_val = pad_batch(val_seqs, max_length=config.MAX_SEQ_LEN, pad_id=config.PAD_ID)
    X_test, M_test = pad_batch(test_seqs, max_length=config.MAX_SEQ_LEN, pad_id=config.PAD_ID)

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    train_loader = Trainer.create_dataloader(X_train, M_train, y_train, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = Trainer.create_dataloader(X_val, M_val, y_val, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = Trainer.create_dataloader(X_test, M_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)

    # 6. Train Classical Baselines
    print("\n[6/8] Training Classical Baselines (TF-IDF + LogReg / SVM)...")
    lr_baseline = TfidfBaseline(model_type="logistic_regression", random_seed=config.RANDOM_SEED)
    lr_baseline.fit(train_df["text"].tolist(), train_df["target"].values)
    lr_metrics = lr_baseline.evaluate(test_df["text"].tolist(), test_df["target"].values)

    svm_baseline = TfidfBaseline(model_type="linear_svm", random_seed=config.RANDOM_SEED)
    svm_baseline.fit(train_df["text"].tolist(), train_df["target"].values)
    svm_metrics = svm_baseline.evaluate(test_df["text"].tolist(), test_df["target"].values)

    print(f"  • TF-IDF + Logistic Regression: Test F1 = {lr_metrics['f1']:.4f}, ROC-AUC = {lr_metrics['roc_auc']:.4f}")
    print(f"  • TF-IDF + Linear SVM:          Test F1 = {svm_metrics['f1']:.4f}, ROC-AUC = {svm_metrics['roc_auc']:.4f}")

    # 7. Train Neural Classifier (Model C)
    print("\n[7/8] Training Neural Classifier (Model C: Trainable Embedding + Masked Pool + Dense + Dropout)...")
    neural_model = build_model_c(
        vocab_size=vocab.vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE,
        pad_id=config.PAD_ID
    )

    param_counts = neural_model.count_parameters()
    print(f"  • Model Parameters: Total = {param_counts['total_params']:,} (Embedding: {param_counts['embedding_params']:,}, Dense: {param_counts['dense_params']:,})")

    checkpoint = ModelCheckpoint(config.BEST_MODEL_PATH, mode="min")
    early_stopping = EarlyStopping(patience=config.PATIENCE, mode="min")
    trainer = Trainer(neural_model, learning_rate=config.LEARNING_RATE)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        verbose=True
    )

    # Evaluate Neural Model on Test Set
    test_probs = trainer.predict_proba(test_loader)
    neural_metrics = compute_classification_metrics(y_test, test_probs)
    neural_metrics["training_time"] = trainer.total_training_time
    neural_metrics["parameters"] = param_counts["total_params"]
    print(f"\n[OK] Neural Classifier Test Evaluation: F1 = {neural_metrics['f1']:.4f}, ROC-AUC = {neural_metrics['roc_auc']:.4f}, Acc = {neural_metrics['accuracy']:.4f}")

    # Save Predictions CSV
    pred_df = test_df.copy()
    pred_df["pred_prob"] = test_probs
    pred_df["pred_label"] = (test_probs >= 0.5).astype(int)
    pred_df["predicted"] = pred_df["pred_label"].map({0: "ham", 1: "spam"})
    pred_df.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"  • Predictions saved to {config.PREDICTIONS_PATH}")

    # Compile Comparison Table
    comparison_data = [
        {
            "model": "Logistic Regression",
            "representation": "TF-IDF (1-2 ngrams)",
            "accuracy": lr_metrics["accuracy"],
            "precision": lr_metrics["precision"],
            "recall": lr_metrics["recall"],
            "f1": lr_metrics["f1"],
            "roc_auc": lr_metrics["roc_auc"],
            "parameters": lr_metrics["num_features"],
            "training_time": lr_metrics["training_time"]
        },
        {
            "model": "Linear SVM",
            "representation": "TF-IDF (1-2 ngrams)",
            "accuracy": svm_metrics["accuracy"],
            "precision": svm_metrics["precision"],
            "recall": svm_metrics["recall"],
            "f1": svm_metrics["f1"],
            "roc_auc": svm_metrics["roc_auc"],
            "parameters": svm_metrics["num_features"],
            "training_time": svm_metrics["training_time"]
        },
        {
            "model": "Neural Classifier (Model C)",
            "representation": "Trainable Embedding (D=64)",
            "accuracy": neural_metrics["accuracy"],
            "precision": neural_metrics["precision"],
            "recall": neural_metrics["recall"],
            "f1": neural_metrics["f1"],
            "roc_auc": neural_metrics["roc_auc"],
            "parameters": param_counts["total_params"],
            "training_time": neural_metrics["training_time"]
        }
    ]
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(config.METRICS_PATH, index=False)
    print(f"  • Metrics summary saved to {config.METRICS_PATH}")

    print("\n" + "=" * 90)
    print(f"{'Model':<28} | {'Representation':<26} | {'F1':<8} | {'ROC-AUC':<8} | {'Params':<8} | {'Time':<8}")
    print("-" * 90)
    for row in comparison_data:
        print(f"{row['model']:<28} | {row['representation']:<26} | {row['f1']:<8.4f} | {row['roc_auc']:<8.4f} | {row['parameters']:<8,d} | {row['training_time']:<8.3f}s")
    print("=" * 90)

    # 8. Error Analysis & Diagnostics
    error_df = ErrorAnalyzer.analyze(
        texts=test_df["text"].tolist(),
        y_true=y_test,
        y_probs=test_probs
    )
    error_df.to_csv(config.ERRORS_PATH, index=False)
    print(f"  • Error diagnostics saved to {config.ERRORS_PATH} ({len(error_df)} misclassified samples)")

    # 9. Generate All 16 Visualizations
    print("\n[8/8] Generating 16 Visualizations...")
    viz = Visualizer(config.CHARTS_DIR)

    # 1-4: Data Distributions
    viz.plot_class_distribution(raw_df)
    viz.plot_message_length_distribution(raw_df)
    token_lens = [len(t) for t in train_tokens]
    viz.plot_sequence_length_distribution(token_lens, max_len=config.MAX_SEQ_LEN)
    viz.plot_vocabulary_frequency(dict(vocab.word_counts))

    # 5-8: Training & Validation Curves
    viz.plot_training_loss(history)
    viz.plot_validation_loss(history)
    viz.plot_training_accuracy(history)
    viz.plot_validation_accuracy(history)

    # 9-11: Performance Curves & Confusion Matrix
    cm_dict = compute_confusion_matrix(y_test, (test_probs >= 0.5).astype(int))
    viz.plot_confusion_matrix(cm_dict)
    viz.plot_roc_curve(y_test, test_probs)
    viz.plot_precision_recall_curve(y_test, test_probs)

    # 12: Model Comparison Bar Chart
    viz.plot_model_comparison(comparison_df)

    # 13: Learned Embedding PCA
    emb_weights = neural_model.embedding.weight.detach().cpu().numpy()
    np.save(config.EMBEDDINGS_DIR / "weights.npy", emb_weights)
    salient_words = [w for w, _ in vocab.word_counts.most_common(40) if w in vocab.word2idx]
    salient_indices = [vocab.word2idx[w] for w in salient_words]
    pca_df = EmbeddingVisualizer.project_pca(emb_weights[salient_indices], salient_words, seed=config.RANDOM_SEED)
    viz.plot_embedding_pca(pca_df)

    # 14-16: Error Analysis Charts
    viz.plot_error_distribution(error_df)
    viz.plot_false_positive_examples(error_df)
    viz.plot_false_negative_examples(error_df)
    print(f"  • Generated all 16 charts in {config.CHARTS_DIR}")

    # Generate Markdown Report
    ReportGenerator.generate(
        comparison_df=comparison_df,
        model_params=param_counts,
        dataset_summary=dataset_summary,
        env_audit=env_audit,
        error_df=error_df,
        output_path=config.REPORT_PATH
    )
    print(f"  • Final evaluation report saved to {config.REPORT_PATH}")

    print("\n[SUCCESS] Day 105 Neural NLP Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
