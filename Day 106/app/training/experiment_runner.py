"""
Experiment runner module for Day 106: RNNs & Sequential Text Learning.
Executes controlled Experiments A, B, C, and D across hyperparameter variations.
"""

from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch

from ..models.experiments import ExperimentConfigs
from ..models.rnn import SimpleRNNClassifier
from ..training.trainer import RNNTrainer
from ..training.callbacks import EarlyStopping
from ..preprocessing.padding import SequencePadder
from ..preprocessing.vocabulary import Vocabulary
from ..preprocessing.encoder import TextEncoder
from ..evaluation.metrics import ClassificationMetrics


class ExperimentRunner:
    """Orchestrates hyperparameter sweep experiments for SimpleRNN."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        vocabulary: Vocabulary,
        random_seed: int = 42
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.vocabulary = vocabulary
        self.encoder = TextEncoder(vocabulary)
        self.random_seed = random_seed

    def _prepare_data(self, seq_len: int):
        train_ids = self.encoder.encode_batch(self.train_df["text"].tolist())
        val_ids = self.encoder.encode_batch(self.val_df["text"].tolist())
        test_ids = self.encoder.encode_batch(self.test_df["text"].tolist())

        train_padded, train_mask = SequencePadder.pad_batch(train_ids, seq_len)
        val_padded, val_mask = SequencePadder.pad_batch(val_ids, seq_len)
        test_padded, test_mask = SequencePadder.pad_batch(test_ids, seq_len)

        train_y = self.train_df["target"].values
        val_y = self.val_df["target"].values
        test_y = self.test_df["target"].values

        train_loader = RNNTrainer.create_dataloader(train_padded, train_mask, train_y, batch_size=32, shuffle=True)
        val_loader = RNNTrainer.create_dataloader(val_padded, val_mask, val_y, batch_size=32, shuffle=False)
        test_loader = RNNTrainer.create_dataloader(test_padded, test_mask, test_y, batch_size=32, shuffle=False)

        return (
            (train_loader, val_loader, test_loader),
            (test_padded, test_mask, test_y)
        )

    def run_single(self, config: Dict[str, Any], epochs: int = 15) -> Dict[str, Any]:
        """Train a single experiment configuration and evaluate on test set."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        seq_len = config["seq_length"]
        loaders, test_data = self._prepare_data(seq_len)
        train_loader, val_loader, test_loader = loaders
        test_padded, test_mask, test_y = test_data

        model = ExperimentConfigs.build_model_from_config(config)
        trainer = RNNTrainer(model, learning_rate=0.001)
        early_stop = EarlyStopping(patience=5)

        start_time = time.perf_counter()
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            early_stopping=early_stop,
            verbose=False
        )
        train_time = time.perf_counter() - start_time

        # Predict on test set
        t_ids = torch.tensor(test_padded, dtype=torch.long)
        t_mask = torch.tensor(test_mask, dtype=torch.float32)
        test_probs = model.predict_proba(t_ids, t_mask)
        metrics = ClassificationMetrics.compute(test_y, test_probs)

        params = model.count_parameters()["total"]
        epochs_run = len(history["train_loss"])

        return {
            "experiment": config["name"],
            "embedding_dim": config["embedding_dim"],
            "hidden_units": config["hidden_units"],
            "sequence_length": config["seq_length"],
            "dropout_1": config.get("dropout_1", 0.3),
            "parameters": params,
            "epochs_run": epochs_run,
            "training_time": round(train_time, 3),
            "train_acc": round(history["train_acc"][-1], 4),
            "val_acc": round(history["val_acc"][-1], 4),
            "test_accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "average_precision": round(metrics["average_precision"], 4),
        }

    def run_all(self, epochs: int = 15) -> pd.DataFrame:
        """Run Experiments A, B, C, and D."""
        vocab_size = len(self.vocabulary)
        configs = [
            ExperimentConfigs.get_experiment_a(vocab_size),
            ExperimentConfigs.get_experiment_b(vocab_size),
            ExperimentConfigs.get_experiment_c(vocab_size),
            ExperimentConfigs.get_experiment_d(vocab_size, seq_length=20),
            ExperimentConfigs.get_experiment_d(vocab_size, seq_length=40),
            ExperimentConfigs.get_experiment_d(vocab_size, seq_length=60),
            ExperimentConfigs.get_experiment_d(vocab_size, seq_length=100),
        ]

        # Deduplicate configs with same name
        seen_names = set()
        unique_configs = []
        for c in configs:
            if c["name"] not in seen_names:
                seen_names.add(c["name"])
                unique_configs.append(c)

        results = []
        for conf in unique_configs:
            res = self.run_single(conf, epochs=epochs)
            results.append(res)

        return pd.DataFrame(results)
