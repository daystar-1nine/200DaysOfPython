"""
Fine-tuning experiments: Experiment A (Frozen BERT) vs Experiment B (Fine-Tuned BERT).
"""
import time
from typing import Dict, Any, Tuple
import pandas as pd
from torch.utils.data import DataLoader
from app.models.bert_classifier import BertSpamClassifier
from app.training.trainer import BertTrainer


def run_experiment_a_frozen(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    model_name: str = "prajjwal1/bert-tiny"
) -> Tuple[BertSpamClassifier, Dict[str, Any], Dict[str, Any]]:
    """
    Experiment A: Feature-based transfer learning.
    BERT backbone is frozen; only the classification head weights are updated.
    """
    print("\n" + "=" * 60)
    print("Starting Experiment A: Frozen BERT Backbone (Classification Head Only)")
    print("=" * 60)

    model = BertSpamClassifier(model_name=model_name, freeze_bert=True)
    param_counts = model.count_parameters()
    print(f"Total Params: {param_counts['total']:,} | Trainable: {param_counts['trainable']:,} | Frozen: {param_counts['frozen']:,}")

    trainer = BertTrainer(model=model, lr=lr)
    start_t = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    duration = time.time() - start_t

    test_metrics = trainer.evaluate(test_loader)
    print(
        f"Experiment A Test Results -> Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['roc_auc']:.4f}"
    )

    summary = {
        "experiment": "Experiment A (Frozen BERT)",
        "trainable_params": param_counts["trainable"],
        "frozen_params": param_counts["frozen"],
        "total_params": param_counts["total"],
        "training_time_seconds": round(duration, 2),
        **{k: test_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "loss"]}
    }

    return model, history, summary


def run_experiment_b_finetuned(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 3,
    lr: float = 2e-5,
    model_name: str = "prajjwal1/bert-tiny"
) -> Tuple[BertSpamClassifier, Dict[str, Any], Dict[str, Any]]:
    """
    Experiment B: End-to-end Fine-Tuning.
    All parameters across the Transformer encoder and classification head are updated with a small learning rate.
    """
    print("\n" + "=" * 60)
    print("Starting Experiment B: End-to-End Fine-Tuning (All Transformer Layers)")
    print("=" * 60)

    model = BertSpamClassifier(model_name=model_name, freeze_bert=False)
    param_counts = model.count_parameters()
    print(f"Total Params: {param_counts['total']:,} | Trainable: {param_counts['trainable']:,} | Frozen: {param_counts['frozen']:,}")

    trainer = BertTrainer(model=model, lr=lr)
    start_t = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    duration = time.time() - start_t

    test_metrics = trainer.evaluate(test_loader)
    print(
        f"Experiment B Test Results -> Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['roc_auc']:.4f}"
    )

    summary = {
        "experiment": "Experiment B (Fine-Tuned BERT)",
        "trainable_params": param_counts["trainable"],
        "frozen_params": param_counts["frozen"],
        "total_params": param_counts["total"],
        "training_time_seconds": round(duration, 2),
        **{k: test_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "loss"]}
    }

    return model, history, summary


def compare_experiments(summary_a: Dict[str, Any], summary_b: Dict[str, Any]) -> pd.DataFrame:
    """Combines Experiment A and B summaries into a clean comparative DataFrame."""
    df = pd.DataFrame([summary_a, summary_b])
    return df
