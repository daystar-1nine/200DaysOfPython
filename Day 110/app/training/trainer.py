"""
Trainer module for Transformer and recurrent classifiers.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from app.training.callbacks import EarlyStopping, ModelCheckpoint
from app.evaluation.metrics import compute_metrics


class Trainer:
    """
    Standard PyTorch training coordinator for sequence classifiers.
    """
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 32,
        patience: int = 4,
        checkpoint_path: Optional[Path] = None
    ) -> Tuple[Dict[str, List[float]], float]:
        """
        Trains model with validation tracking, early stopping, and timing.
        """
        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_ds = TensorDataset(
            torch.tensor(x_val, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=patience, mode="min")
        checkpoint = ModelCheckpoint(checkpoint_path) if checkpoint_path else None

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": []
        }

        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(batch_x)
                probs = out[0] if isinstance(out, tuple) else out
                loss = self.criterion(probs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * len(batch_y)
                preds = (probs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

            train_loss /= max(total, 1)
            train_acc = correct / max(total, 1)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds_list = []
            val_probs_list = []
            val_targets_list = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    out = self.model(batch_x)
                    probs = out[0] if isinstance(out, tuple) else out
                    loss = self.criterion(probs, batch_y)
                    val_loss += loss.item() * len(batch_y)

                    val_probs_list.extend(probs.cpu().numpy().ravel())
                    val_targets_list.extend(batch_y.cpu().numpy().ravel())

            val_loss /= max(len(val_targets_list), 1)
            val_probs_arr = np.array(val_probs_list)
            val_targets_arr = np.array(val_targets_list)
            val_metrics = compute_metrics(val_targets_arr, val_probs_arr)

            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_metrics["accuracy"]))
            history["val_f1"].append(float(val_metrics["f1"]))

            if checkpoint:
                checkpoint(val_loss, self.model)

            if early_stopping(val_loss, self.model):
                early_stopping.restore(self.model)
                break

        elapsed_time = time.time() - start_time
        return history, elapsed_time

    def predict_proba(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Computes prediction probabilities for given input sequences."""
        self.model.eval()
        ds = TensorDataset(torch.tensor(x, dtype=torch.long))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probs_all = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                out = self.model(batch_x)
                probs = out[0] if isinstance(out, tuple) else out
                probs_all.extend(probs.cpu().numpy().ravel())

        return np.array(probs_all)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)
