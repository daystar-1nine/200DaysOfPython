"""
Experiment 2: Embedding Dimension Sensitivity Study (D in [16, 32, 64, 128]).
Day 105: Neural NLP & Text Classification.
"""

import sys
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import config
from app.data.loader import DataLoader
from app.data.splitter import DataSplitter
from app.preprocessing.tokenizer import Tokenizer
from app.preprocessing.vocabulary import Vocabulary
from app.preprocessing.encoder import TextEncoder
from app.preprocessing.padding import pad_batch
from app.models.architectures import build_model_c
from app.training.trainer import Trainer
from app.training.callbacks import EarlyStopping
from app.evaluation.metrics import compute_classification_metrics


def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 2: EMBEDDING DIMENSION SENSITIVITY STUDY")
    print("=" * 80)

    raw_df = DataLoader.load_raw_data(config.RAW_DATA_PATH)
    encoded_df = DataLoader.encode_labels(raw_df)
    train_df, val_df, test_df = DataSplitter.split(encoded_df, random_seed=42)

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

    dims = [16, 32, 64, 128]
    records = []

    for d in dims:
        model = build_model_c(vocab.vocab_size, embedding_dim=d, hidden_dim=64, dropout_rate=0.5)
        params = model.count_parameters()
        trainer = Trainer(model, learning_rate=0.002)
        history = trainer.fit(train_loader, val_loader, epochs=20, early_stopping=EarlyStopping(patience=5), verbose=False)

        val_probs = trainer.predict_proba(val_loader)
        val_m = compute_classification_metrics(val_df["target"].values, val_probs)

        test_probs = trainer.predict_proba(test_loader)
        test_m = compute_classification_metrics(test_df["target"].values, test_probs)

        records.append({
            "Embedding Dim": d,
            "Total Params": f"{params['total_params']:,}",
            "Embedding Params": f"{params['embedding_params']:,}",
            "Val Loss": f"{min(history['val_loss']):.4f}",
            "Val F1": f"{val_m['f1']:.4f}",
            "Test F1": f"{test_m['f1']:.4f}",
            "Test ROC-AUC": f"{test_m['roc_auc']:.4f}",
            "Train Time": f"{trainer.total_training_time:.2f}s"
        })

    df = pd.DataFrame(records)
    print("\nResults across Embedding Dimensions:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
