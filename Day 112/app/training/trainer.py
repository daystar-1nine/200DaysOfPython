"""
Training engine for MiniGPT with teacher forcing, gradient clipping, and perplexity tracking.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import pandas as pd
import torch
import torch.nn as nn
from app.data.batching import get_batch
from app.evaluation.metrics import compute_perplexity


class MiniGPTTrainer:
    """
    Orchestrates MiniGPT training, validation evaluation, gradient clipping,
    and metrics collection across training iterations.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        context_length: int = 64,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.grad_clip = grad_clip

    @torch.no_grad()
    def estimate_loss(
        self,
        data: torch.Tensor,
        eval_iters: int = 20
    ) -> float:
        """Computes average loss across multiple randomly sampled evaluation batches."""
        self.model.eval()
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, self.batch_size, self.context_length, device=self.device)
            _, loss, _ = self.model(x, targets=y)
            losses.append(loss.item())
        return float(sum(losses) / max(len(losses), 1))

    def train(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        max_iters: int = 600,
        eval_interval: int = 100,
        eval_iters: int = 20,
        output_csv: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict[str, List[Any]]:
        """
        Executes full training iteration loop and outputs metrics tracking history.
        """
        history = {
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "perplexity": [],
            "learning_rate": []
        }

        start_time = time.time()
        for iter_num in range(1, max_iters + 1):
            self.model.train()
            x, y = get_batch(train_data, self.batch_size, self.context_length, device=self.device)

            logits, loss, _ = self.model(x, targets=y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Periodic validation evaluation
            if iter_num % eval_interval == 0 or iter_num == 1 or iter_num == max_iters:
                train_loss_est = self.estimate_loss(train_data, eval_iters=eval_iters)
                val_loss_est = self.estimate_loss(val_data, eval_iters=eval_iters)
                ppl = compute_perplexity(val_loss_est)
                lr = self.optimizer.param_groups[0]["lr"]

                history["step"].append(iter_num)
                history["train_loss"].append(round(train_loss_est, 4))
                history["val_loss"].append(round(val_loss_est, 4))
                history["perplexity"].append(round(ppl, 2))
                history["learning_rate"].append(lr)

                if verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"Step {iter_num:04d}/{max_iters:04d} | "
                        f"Train Loss: {train_loss_est:.4f} | "
                        f"Val Loss: {val_loss_est:.4f} | "
                        f"Perplexity: {ppl:.2f} | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(history)
            df.to_csv(output_csv, index=False)
            if verbose:
                print(f"Metrics history saved to: {output_csv}")

        return history
