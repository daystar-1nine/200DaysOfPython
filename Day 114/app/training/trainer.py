"""
SFTTrainer: Supervised Fine-Tuning execution engine with gradient accumulation,
learning rate schedules, loss masking, and telemetry logging.
"""
import time
import math
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from app.config import SFTTrainingConfig, ChatModelConfig
from app.data.tokenizer import ChatTokenizer
from app.data.collator import SFTDataCollator
from app.training.losses import compute_masked_loss, compute_token_accuracy
from app.training.checkpoint import save_sft_checkpoint


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


class SFTTrainer:
    """
    Executes instruction fine-tuning on conversation datasets using masked next-token loss.
    """
    def __init__(
        self,
        model: nn.Module,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        tokenizer: ChatTokenizer,
        config: Optional[SFTTrainingConfig] = None,
        device: Optional[torch.device] = None,
        output_dir: Optional[Union[str, Path]] = None,
        experiment_name: str = "sft_run"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config or SFTTrainingConfig()
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints" / experiment_name
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.csv_log_path = self.metrics_dir / f"{experiment_name}.csv"

        # Collator
        self.collator = SFTDataCollator(
            tokenizer=tokenizer,
            max_length=getattr(self.model.config, "context_length", 128),
            mask_prompt_loss=self.config.mask_prompt_loss
        )

        # DataLoaders
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

        # Setup Optimizer with weight decay separation
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "ln" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        # Setup Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = min(self.config.warmup_steps, max(1, int(total_steps * 0.1)))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
            min_lr_ratio=self.config.min_learning_rate / self.config.learning_rate
        )

        self.history: List[Dict[str, Any]] = []
        self._init_csv()

    def _init_csv(self) -> None:
        """Initializes the CSV metrics log file."""
        with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "step", "train_loss", "train_ppl", "train_acc",
                "val_loss", "val_ppl", "val_acc", "lr", "tokens_per_sec", "elapsed_s"
            ])

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluates loss, perplexity, and token accuracy on validation split."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _ = self.model(input_ids, attention_mask=attention_mask)
            loss, _, num_active = compute_masked_loss(logits, labels)
            acc = compute_token_accuracy(logits, labels)

            total_loss += loss.item() * max(1, num_active)
            total_tokens += max(1, num_active)
            total_correct += acc * max(1, num_active)

        mean_loss = total_loss / max(1, total_tokens)
        perplexity = round(math.exp(min(mean_loss, 20.0)), 4)
        accuracy = round(total_correct / max(1, total_tokens), 4)

        return {
            "val_loss": round(mean_loss, 4),
            "val_ppl": perplexity,
            "val_acc": accuracy
        }

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """Executes full SFT training loop across specified epochs."""
        global_step = 0
        best_val_loss = float("inf")
        start_time = time.time()
        total_tokens_seen = 0

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0

            for batch_idx, batch in enumerate(self.train_loader, start=1):
                step_start = time.time()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, _ = self.model(input_ids, attention_mask=attention_mask)
                loss, ppl, num_active = compute_masked_loss(logits, labels)
                acc = compute_token_accuracy(logits, labels)

                # Micro-batch gradient scaling
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()

                epoch_loss += loss.item() * max(1, num_active)
                epoch_tokens += max(1, num_active)
                total_tokens_seen += input_ids.numel()

                if batch_idx % self.config.gradient_accumulation_steps == 0 or batch_idx == len(self.train_loader):
                    if self.config.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                step_duration = max(time.time() - step_start, 1e-4)
                tokens_per_sec = round(input_ids.numel() / step_duration, 1)
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Periodic evaluation
                if global_step > 0 and (global_step % self.config.eval_interval == 0 or batch_idx == len(self.train_loader)):
                    eval_metrics = self.evaluate()
                    self.model.train()

                    val_loss = eval_metrics["val_loss"]
                    val_ppl = eval_metrics["val_ppl"]
                    val_acc = eval_metrics["val_acc"]

                    # Checkpoint best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_sft_checkpoint(
                            self.checkpoints_dir / "best_model.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            step=global_step,
                            metrics=eval_metrics
                        )

                    log_entry = {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": round(loss.item(), 4),
                        "train_ppl": ppl,
                        "train_acc": acc,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "val_acc": val_acc,
                        "lr": round(current_lr, 7),
                        "tokens_per_sec": tokens_per_sec,
                        "elapsed_s": round(time.time() - start_time, 2)
                    }
                    self.history.append(log_entry)

                    with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            log_entry["epoch"], log_entry["step"],
                            log_entry["train_loss"], log_entry["train_ppl"], log_entry["train_acc"],
                            log_entry["val_loss"], log_entry["val_ppl"], log_entry["val_acc"],
                            f"{log_entry['lr']:.7e}", log_entry["tokens_per_sec"], log_entry["elapsed_s"]
                        ])

                    if verbose and global_step % self.config.eval_interval == 0:
                        print(f"Epoch {epoch:02d} | Step {global_step:04d} | "
                              f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
                              f"Val PPL: {val_ppl:.2f} | Val Acc: {val_acc*100:.1f}% | TPS: {tokens_per_sec}")

        # Final checkpoint
        save_sft_checkpoint(
            self.checkpoints_dir / "final_model.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.config.num_epochs,
            step=global_step,
            metrics={"best_val_loss": best_val_loss}
        )

        return {
            "total_steps": global_step,
            "best_val_loss": best_val_loss,
            "total_tokens_seen": total_tokens_seen,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "history": self.history
        }
