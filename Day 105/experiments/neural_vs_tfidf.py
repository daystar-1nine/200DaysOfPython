"""
Experiment 1: Classical NLP (TF-IDF + LogReg / SVM) vs Neural Classifier.
Day 105: Neural NLP & Text Classification.
"""

import sys
from pathlib import Path
import pandas as pd

# Safe stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add app parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import config
from app.data.loader import DataLoader
from app.data.splitter import DataSplitter
from app.preprocessing.tokenizer import Tokenizer
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.encoder import TextEncoder
from app.preprocessing.padding import pad_batch
from app.models.baseline import TfidfBaseline
from app.models.architectures import build_model_c
from app.training.trainer import Trainer
from app.training.callbacks import EarlyStopping
from app.evaluation.metrics import compute_classification_metrics


def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 1: CLASSICAL (TF-IDF) VS NEURAL TEXT CLASSIFICATION")
    print("=" * 80)

    # 1. Prepare data
    raw_df = DataLoader.load_raw_data(config.RAW_DATA_PATH)
    encoded_df = DataLoader.encode_labels(raw_df)
    train_df, val_df, test_df = DataSplitter.split(encoded_df, random_seed=42)

    # 2. Classical Baselines
    print("\n[1/3] Training Logistic Regression on TF-IDF...")
    lr = TfidfBaseline(model_type="logistic_regression", random_seed=42)
    lr.fit(train_df["text"].tolist(), train_df["target"].values)
    lr_metrics = lr.evaluate(test_df["text"].tolist(), test_df["target"].values)

    print("[2/3] Training Linear SVM on TF-IDF...")
    svm = TfidfBaseline(model_type="linear_svm", random_seed=42)
    svm.fit(train_df["text"].tolist(), train_df["target"].values)
    svm_metrics = svm.evaluate(test_df["text"].tolist(), test_df["target"].values)

    # 3. Neural Classifier
    print("[3/3] Training Neural Classifier (Model C)...")
    tokenizer = Tokenizer()
    train_tokens = [tokenizer.tokenize(t) for t in train_df["text"]]
    vocab = Vocabulary(pad_id=0, unk_id=1).fit(train_tokens, min_freq=2)
    encoder = TextEncoder(tokenizer, vocab)

    X_train, M_train = pad_batch(encoder.encode_corpus(train_df["text"].tolist()), max_length=40)
    X_val, M_val = pad_batch(encoder.encode_corpus(val_df["text"].tolist()), max_length=40)
    X_test, M_test = pad_batch(encoder.encode_corpus(test_df["text"].tolist()), max_length=40)

    train_loader = Trainer.create_dataloader(X_train, M_train, train_df["target"].values, batch_size=32)
    val_loader = Trainer.create_dataloader(X_val, M_val, val_df["target"].values, batch_size=32, shuffle=False)
    test_loader = Trainer.create_dataloader(X_test, M_test, test_df["target"].values, batch_size=32, shuffle=False)

    model = build_model_c(vocab.vocab_size, embedding_dim=64, hidden_dim=64, dropout_rate=0.5)
    trainer = Trainer(model, learning_rate=0.002)
    trainer.fit(train_loader, val_loader, epochs=20, early_stopping=EarlyStopping(patience=5), verbose=False)

    neural_probs = trainer.predict_proba(test_loader)
    neural_metrics = compute_classification_metrics(test_df["target"].values, neural_probs)

    # Results Table
    results = [
        {"Model": "Logistic Regression", "Rep": "TF-IDF", "Accuracy": lr_metrics["accuracy"], "F1": lr_metrics["f1"], "ROC-AUC": lr_metrics["roc_auc"], "Time": f"{lr_metrics['training_time']:.3f}s"},
        {"Model": "Linear SVM", "Rep": "TF-IDF", "Accuracy": svm_metrics["accuracy"], "F1": svm_metrics["f1"], "ROC-AUC": svm_metrics["roc_auc"], "Time": f"{svm_metrics['training_time']:.3f}s"},
        {"Model": "Neural Classifier", "Rep": "Trainable Emb (D=64)", "Accuracy": neural_metrics["accuracy"], "F1": neural_metrics["f1"], "ROC-AUC": neural_metrics["roc_auc"], "Time": f"{trainer.total_training_time:.3f}s"}
    ]
    df_res = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
