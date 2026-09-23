"""
Experiment 3: Sequence Length Sensitivity Study (T in [20, 40, 60, 100]).
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
    print("EXPERIMENT 3: MAXIMUM SEQUENCE LENGTH STUDY")
    print("=" * 80)

    raw_df = DataLoader.load_raw_data(config.RAW_DATA_PATH)
    encoded_df = DataLoader.encode_labels(raw_df)
    train_df, val_df, test_df = DataSplitter.split(encoded_df, random_seed=42)

    tokenizer = Tokenizer()
    train_tokens = [tokenizer.tokenize(t) for t in train_df["text"]]
    vocab = Vocabulary(pad_id=0, unk_id=1).fit(train_tokens, min_freq=2)
    encoder = TextEncoder(tokenizer, vocab)

    all_encoded_train = encoder.encode_corpus(train_df["text"].tolist())
    all_encoded_val = encoder.encode_corpus(val_df["text"].tolist())
    all_encoded_test = encoder.encode_corpus(test_df["text"].tolist())

    seq_lengths = [20, 40, 60, 100]
    records = []

    for t_len in seq_lengths:
        # Calculate truncation percentage on train set
        trunc_count = sum(1 for s in all_encoded_train if len(s) > t_len)
        trunc_pct = (trunc_count / len(all_encoded_train)) * 100

        X_train, M_train = pad_batch(all_encoded_train, max_length=t_len)
        X_val, M_val = pad_batch(all_encoded_val, max_length=t_len)
        X_test, M_test = pad_batch(all_encoded_test, max_length=t_len)

        train_loader = Trainer.create_dataloader(X_train, M_train, train_df["target"].values, batch_size=32)
        val_loader = Trainer.create_dataloader(X_val, M_val, val_df["target"].values, batch_size=32, shuffle=False)
        test_loader = Trainer.create_dataloader(X_test, M_test, test_df["target"].values, batch_size=32, shuffle=False)

        model = build_model_c(vocab.vocab_size, embedding_dim=64, hidden_dim=64, dropout_rate=0.5)
        trainer = Trainer(model, learning_rate=0.002)
        history = trainer.fit(train_loader, val_loader, epochs=20, early_stopping=EarlyStopping(patience=5), verbose=False)

        test_probs = trainer.predict_proba(test_loader)
        test_m = compute_classification_metrics(test_df["target"].values, test_probs)

        records.append({
            "Max Seq Len": t_len,
            "Truncated Train": f"{trunc_pct:.1f}%",
            "Val Loss": f"{min(history['val_loss']):.4f}",
            "Test F1": f"{test_m['f1']:.4f}",
            "Test ROC-AUC": f"{test_m['roc_auc']:.4f}",
            "Train Time": f"{trainer.total_training_time:.2f}s"
        })

    df = pd.DataFrame(records)
    print("\nResults across Maximum Sequence Lengths:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
