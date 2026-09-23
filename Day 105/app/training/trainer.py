"""
Trainer module for Day 105: Neural NLP & Text Classification.
Executes mini-batch optimization, tracks learning curves, and coordinates callbacks.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .callbacks import EarlyStopping
from .checkpointing import ModelCheckpoint


class Trainer:
    """Trains a PyTorch neural text classification model with evaluation tracking."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.total_training_time: float = 0.0

    @staticmethod
    def create_dataloader(
        x: np.ndarray,
        masks: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch DataLoader from NumPy sequence arrays."""
        x_tensor = torch.tensor(x, dtype=torch.long)
        mask_tensor = torch.tensor(masks, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(x_tensor, mask_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for x_batch, mask_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch, mask=mask_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate loss and accuracy on a validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, mask_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch, mask=mask_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total_samples += len(y_batch)

        return (total_loss / total_samples), (correct / total_samples)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 25,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Execute full training loop over epochs."""
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                )

            # Checkpoint best model
            if checkpoint is not None:
                checkpoint.step(val_loss, self.model)

            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_loss):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch:02d}.")
                    break

        self.total_training_time = time.perf_counter() - start_time
        if verbose:
            print(f"Training completed in {self.total_training_time:.2f}s.")

        # Restore best checkpoint if available
        if checkpoint is not None and checkpoint.filepath.exists():
            checkpoint.load_best(self.model)

        return self.history

    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """Generate predicted probabilities across a DataLoader."""
        self.model.eval()
        probs_list = []
        with torch.no_grad():
            for x_batch, mask_batch, _ in dataloader:
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                probs = self.model.predict_proba(x_batch, mask=mask_batch)
                probs_list.extend(probs.cpu().numpy().tolist())
        return np.array(probs_list, dtype=np.float32)
