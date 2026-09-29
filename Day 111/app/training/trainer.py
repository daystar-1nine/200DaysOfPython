"""
Trainer module for training, evaluating, and extracting representations from BERT models.
"""
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.evaluation.metrics import compute_metrics


class BertTrainer:
    """
    Standard training and evaluation harness for PyTorch BERT classifiers.
    """
    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()

        # Optimize only parameters requiring gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs one epoch of model training."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        start_time = time.time()

        for batch in dataloader:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            token_type_ids = batch[2].to(self.device)
            labels = batch[3].to(self.device)

            self.optimizer.zero_grad()
            probs, _, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Clamp probabilities for numerical stability in BCELoss
            probs_clamped = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
            loss = self.criterion(probs_clamped, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_samples += batch_size

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        elapsed = time.time() - start_time

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "time_seconds": elapsed
        }

    def evaluate(self, dataloader: DataLoader, threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluates the model on validation or test data."""
        self.model.eval()
        total_loss = 0.0
        all_probs: List[float] = []
        all_labels: List[int] = []
        start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                token_type_ids = batch[2].to(self.device)
                labels = batch[3].to(self.device)

                probs, _, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                probs_clamped = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
                loss = self.criterion(probs_clamped, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size

                all_probs.extend(probs.squeeze(-1).cpu().numpy().tolist())
                all_labels.extend(labels.squeeze(-1).cpu().numpy().astype(int).tolist())

        elapsed = time.time() - start_time
        total_samples = len(all_labels)
        avg_loss = total_loss / max(total_samples, 1)

        y_probs = np.array(all_probs)
        y_preds = (y_probs >= threshold).astype(int)
        y_true = np.array(all_labels)

        metrics = compute_metrics(y_true, y_preds, y_probs)
        metrics["loss"] = round(avg_loss, 4)
        metrics["eval_time_seconds"] = round(elapsed, 4)
        metrics["y_true"] = y_true.tolist()
        metrics["y_pred"] = y_preds.tolist()
        metrics["y_probs"] = y_probs.tolist()
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 3,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Executes complete training across multiple epochs and tracks metrics history.
        """
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "time_per_epoch": []
        }

        best_val_f1 = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_res = self.train_epoch(train_loader)
            val_res = self.evaluate(val_loader)

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_res["loss"], 4))
            history["train_acc"].append(round(train_res["accuracy"], 4))
            history["val_loss"].append(round(val_res["loss"], 4))
            history["val_acc"].append(round(val_res["accuracy"], 4))
            history["val_f1"].append(round(val_res["f1"], 4))
            history["time_per_epoch"].append(round(train_res["time_seconds"], 2))

            if val_res["f1"] > best_val_f1:
                best_val_f1 = val_res["f1"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_res['loss']:.4f} Acc: {train_res['accuracy']:.4f} | "
                    f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']:.4f} F1: {val_res['f1']:.4f} | "
                    f"Time: {train_res['time_seconds']:.2f}s"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def extract_representations(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts [CLS] representations, predicted probabilities, and ground truth labels.
        Returns:
            cls_embeddings: (N, hidden_size)
            probs: (N,)
            labels: (N,)
        """
        self.model.eval()
        all_cls = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                token_type_ids = batch[2].to(self.device)
                labels = batch[3]

                probs, cls_repr, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                all_cls.append(cls_repr.cpu().numpy())
                all_probs.append(probs.squeeze(-1).cpu().numpy())
                all_labels.append(labels.squeeze(-1).numpy())

        return (
            np.vstack(all_cls),
            np.concatenate(all_probs),
            np.concatenate(all_labels)
        )
