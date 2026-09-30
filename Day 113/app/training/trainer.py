"""
ScalingTrainer: Core training engine for LLM scaling experiments.
Features:
- Micro-batch gradient accumulation (effective batch scaling)
- Learning rate schedule integration
- Gradient clipping
- Throughput (tokens/sec) and memory tracking
- Periodic validation loss and perplexity evaluation
"""
import time
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import psutil
import torch
import torch.nn as nn
from app.training.scheduler import configure_scheduler
from app.training.checkpoint import save_checkpoint
from app.training.logger import TrainingCSVLogger
from app.config import TrainingConfig


def sample_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples random sequences with shifted teacher-forcing targets:
      x = data[i : i + context_length]
      y = data[i + 1 : i + 1 + context_length]
    """
    if len(data) <= context_length + 1:
        raise ValueError(f"Data length ({len(data)}) must be strictly greater than context_length + 1 ({context_length + 1})")

    max_idx = len(data) - context_length - 1
    ix = torch.randint(0, max_idx + 1, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix]).to(device)
    return x, y


class ScalingTrainer:
    """
    Orchestrates training for ScalableMiniGPT across scaling experiments.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
        logger: Optional[TrainingCSVLogger] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.config = config or TrainingConfig()
        self.scheduler = scheduler or configure_scheduler(
            optimizer,
            scheduler_type=self.config.scheduler_type,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps
        )
        self.logger = logger
        self.process = psutil.Process(os.getpid())

    def get_memory_mb(self) -> float:
        """Returns current process RAM usage in MB."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            return round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2)
        return round(self.process.memory_info().rss / (1024 * 1024), 2)

    @torch.no_grad()
    def estimate_loss(
        self,
        data: torch.Tensor,
        context_length: int,
        eval_iters: int = 10,
        batch_size: int = 16
    ) -> Dict[str, float]:
        """Evaluates average cross-entropy loss and perplexity on dataset."""
        self.model.eval()
        losses = []
        for _ in range(eval_iters):
            x, y = sample_batch(data, batch_size=batch_size, context_length=context_length, device=self.device)
            _, loss, _ = self.model(x, targets=y)
            losses.append(loss.item())

        mean_loss = float(sum(losses) / max(len(losses), 1))
        # Perplexity = exp(loss)
        clamped_loss = min(mean_loss, 50.0)
        perplexity = round(math.exp(clamped_loss), 4)

        return {
            "loss": round(mean_loss, 4),
            "perplexity": perplexity
        }

    def train(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        context_length: int = 64,
        checkpoint_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Executes pretraining loop with gradient accumulation, clipping,
        and throughput profiling.
        """
        self.model.train()
        cfg = self.config
        grad_accum = max(1, cfg.gradient_accumulation_steps)
        effective_batch = cfg.batch_size
        micro_batch = max(1, effective_batch // grad_accum)

        total_tokens_seen = 0
        start_time = time.time()
        last_val_loss = None
        last_val_ppl = None

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, cfg.max_steps + 1):
            self.optimizer.zero_grad()
            step_loss = 0.0

            # Gradient Accumulation Loop across micro-batches
            for _ in range(grad_accum):
                x, y = sample_batch(train_data, batch_size=micro_batch, context_length=context_length, device=self.device)
                logits, loss, _ = self.model(x, targets=y)
                # Scale loss by accumulation steps
                scaled_loss = loss / grad_accum
                scaled_loss.backward()
                step_loss += loss.item() / grad_accum
                total_tokens_seen += (micro_batch * context_length)

            # Gradient Clipping
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)

            # Optimizer and Scheduler Step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            elapsed = max(0.001, time.time() - start_time)
            tps = total_tokens_seen / elapsed
            current_lr = self.optimizer.param_groups[0]["lr"]
            current_mem = self.get_memory_mb()

            # Evaluation Interval
            if step % cfg.eval_interval == 0 or step == 1 or step == cfg.max_steps:
                val_metrics = self.estimate_loss(
                    val_data,
                    context_length=context_length,
                    eval_iters=cfg.eval_iters,
                    batch_size=micro_batch
                )
                last_val_loss = val_metrics["loss"]
                last_val_ppl = val_metrics["perplexity"]

                if verbose:
                    print(
                        f"Step {step:04d}/{cfg.max_steps:04d} | "
                        f"Train Loss: {step_loss:.4f} | "
                        f"Val Loss: {last_val_loss:.4f} | "
                        f"Val PPL: {last_val_ppl:>6.2f} | "
                        f"TPS: {tps:>6.1f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Mem: {current_mem:.1f} MB"
                    )

                if checkpoint_dir:
                    save_checkpoint(
                        filepath=checkpoint_dir / f"step_{step:04d}.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=step,
                        metrics={"train_loss": step_loss, "val_loss": last_val_loss, "val_ppl": last_val_ppl}
                    )

            if self.logger:
                self.logger.log(
                    step=step,
                    loss=step_loss,
                    val_loss=last_val_loss,
                    val_ppl=last_val_ppl,
                    learning_rate=current_lr,
                    tokens_seen=total_tokens_seen,
                    tokens_per_sec=tps,
                    memory_mb=current_mem,
                    elapsed_time=elapsed
                )

        total_time = time.time() - start_time
        final_val = self.estimate_loss(val_data, context_length=context_length, eval_iters=cfg.eval_iters)

        return {
            "final_train_loss": round(step_loss, 4),
            "final_val_loss": final_val["loss"],
            "final_val_ppl": final_val["perplexity"],
            "total_tokens_seen": total_tokens_seen,
            "total_time_seconds": round(total_time, 2),
            "average_tokens_per_second": round(total_tokens_seen / max(total_time, 0.001), 2),
            "peak_memory_mb": current_mem
        }
