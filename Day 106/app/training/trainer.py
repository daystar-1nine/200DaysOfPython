"""
Model trainer module for Day 106: RNNs & Sequential Text Learning.
Executes training and validation loops with gradient clipping, early stopping, and history tracking.
"""

import time
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .callbacks import EarlyStopping, ModelCheckpoint


class RNNTrainer:
    """Trains PyTorch RNN models with Binary Cross-Entropy loss and Adam optimizer."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        grad_clip: float = 5.0,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.grad_clip = grad_clip
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    @staticmethod
    def create_dataloader(
        input_ids: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader from NumPy arrays."""
        t_ids = torch.tensor(input_ids, dtype=torch.long)
        t_masks = torch.tensor(masks, dtype=torch.float32)
        t_labels = torch.tensor(labels, dtype=torch.float32)
        dataset = TensorDataset(t_ids, t_masks, t_labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_ids, batch_masks, batch_labels in dataloader:
            batch_ids = batch_ids.to(self.device)
            batch_masks = batch_masks.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_ids, batch_masks)
            loss = self.criterion(logits, batch_labels)
            loss.backward()

            # Prevent exploding gradients via gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * batch_ids.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total_samples += batch_ids.size(0)

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Evaluate model loss and accuracy on validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_ids, batch_masks, batch_labels in dataloader:
                batch_ids = batch_ids.to(self.device)
                batch_masks = batch_masks.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_ids, batch_masks)
                loss = self.criterion(logits, batch_labels)

                total_loss += loss.item() * batch_ids.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == batch_labels).sum().item()
                total_samples += batch_ids.size(0)

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        return epoch_loss, epoch_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Execute full training pipeline."""
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self.train_epoch(train_loader)
            v_loss, v_acc = self.evaluate(val_loader)

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_acc"].append(v_acc)

            if checkpoint is not None:
                checkpoint.step(v_loss, self.model)

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.4f} | "
                    f"Val Loss: {v_loss:.4f} - Val Acc: {v_acc:.4f}"
                )

            if early_stopping is not None and early_stopping(v_loss, self.model):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                early_stopping.restore_best_weights(self.model)
                break

        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"Training completed in {elapsed:.2f}s.")
        return self.history
