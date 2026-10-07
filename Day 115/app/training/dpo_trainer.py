"""
DPOTrainer engine for Day 115: Preference Optimization & RLHF.
Executes Direct Preference Optimization (DPO) on pairwise conversational datasets,
guaranteeing reference model isolation while tracking policy loss, implicit rewards,
and preference accuracy.
"""
import time
import math
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from app.config import DPOTrainingConfig
from app.data.tokenizer import ChatTokenizer
from app.data.collator import PreferenceCollator
from app.models.dpo_model import DPOModel
from app.training.checkpoint import save_checkpoint


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """Cosine decay schedule with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class DPOTrainer:
    """
    Executes Direct Preference Optimization (DPO) training and validation.
    """
    def __init__(
        self,
        dpo_model: DPOModel,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        tokenizer: ChatTokenizer,
        config: Optional[DPOTrainingConfig] = None,
        output_dir: Optional[Path] = None,
        experiment_name: str = "dpo_standard"
    ):
        self.config = config or DPOTrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dpo_model = dpo_model.to(self.device)
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data

        self.output_dir = Path(output_dir or "outputs")
        self.metrics_dir = self.output_dir / "metrics"
        self.checkpoints_dir = self.output_dir / "checkpoints" / experiment_name
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.collator = PreferenceCollator(tokenizer)
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator
        )

        # Optimize ONLY policy parameters; reference parameters remain frozen
        self.optimizer = torch.optim.AdamW(
            self.dpo_model.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=self.config.min_learning_rate / self.config.learning_rate
        )

        self.history: List[Dict[str, Any]] = []

    def evaluate(self) -> Dict[str, float]:
        """Runs evaluation over the validation set."""
        self.dpo_model.policy.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_margin = 0.0
        total_rc = 0.0
        total_rr = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                device_batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, metrics = self.dpo_model.forward_pairwise(
                    device_batch,
                    beta=self.config.beta,
                    loss_type=self.config.loss_type
                )

                total_loss += metrics["dpo_loss"]
                total_acc += metrics["accuracy"]
                total_margin += metrics["reward_margin"]
                total_rc += metrics["chosen_reward"]
                total_rr += metrics["rejected_reward"]
                count += 1

        return {
            "val_loss": round(total_loss / max(1, count), 4),
            "val_accuracy": round((total_acc / max(1, count)) * 100, 2),
            "val_margin": round(total_margin / max(1, count), 4),
            "val_chosen_reward": round(total_rc / max(1, count), 4),
            "val_rejected_reward": round(total_rr / max(1, count), 4)
        }

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """Main DPO training loop."""
        start_time = time.time()
        best_val_acc = -1.0
        global_step = 0

        for epoch in range(1, self.config.num_epochs + 1):
            self.dpo_model.policy.train()
            self.dpo_model.reference.eval()

            for step, batch in enumerate(self.train_loader, start=1):
                global_step += 1
                device_batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                loss, metrics = self.dpo_model.forward_pairwise(
                    device_batch,
                    beta=self.config.beta,
                    loss_type=self.config.loss_type
                )

                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.dpo_model.policy.parameters(), self.config.grad_clip)

                self.optimizer.step()
                self.scheduler.step()

                # Periodic evaluation
                if global_step % self.config.eval_interval == 0 or global_step == 1:
                    val_metrics = self.evaluate()
                    self.dpo_model.policy.train()

                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    log_record = {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": round(metrics["dpo_loss"], 4),
                        "train_acc": round(metrics["accuracy"] * 100, 2),
                        "train_margin": round(metrics["reward_margin"], 4),
                        "val_loss": val_metrics["val_loss"],
                        "val_acc": val_metrics["val_accuracy"],
                        "val_margin": val_metrics["val_margin"],
                        "chosen_reward": val_metrics["val_chosen_reward"],
                        "rejected_reward": val_metrics["val_rejected_reward"],
                        "learning_rate": cur_lr
                    }
                    self.history.append(log_record)

                    if verbose:
                        print(f"Epoch {epoch:02d} | Step {global_step:04d} | Train Loss: {metrics['dpo_loss']:.4f} | "
                              f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']:.1f}% | "
                              f"Margin: {val_metrics['val_margin']:.3f}")

                    if val_metrics["val_accuracy"] > best_val_acc:
                        best_val_acc = val_metrics["val_accuracy"]
                        save_checkpoint(
                            self.checkpoints_dir / "best_model.pt",
                            self.dpo_model.policy,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            step=global_step,
                            metrics=val_metrics
                        )

        # Final evaluation and checkpoint
        final_val = self.evaluate()
        save_checkpoint(
            self.checkpoints_dir / "final_model.pt",
            self.dpo_model.policy,
            optimizer=self.optimizer,
            epoch=self.config.num_epochs,
            step=global_step,
            metrics=final_val
        )

        # Export metrics CSV
        csv_path = self.metrics_dir / f"{self.experiment_name}.csv"
        if self.history:
            keys = self.history[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)

        elapsed = round(time.time() - start_time, 2)
        return {
            "total_steps": global_step,
            "final_val_loss": final_val["val_loss"],
            "final_val_accuracy": final_val["val_accuracy"],
            "final_val_margin": final_val["val_margin"],
            "best_val_accuracy": best_val_acc,
            "elapsed_seconds": elapsed,
            "csv_path": str(csv_path)
        }
