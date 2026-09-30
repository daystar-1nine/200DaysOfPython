"""
Lightweight CSV experiment logger for tracking training loss, validation perplexity,
throughput (tokens/sec), learning rate, and memory consumption.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
import pandas as pd


class TrainingCSVLogger:
    """Logs per-step training metrics incrementally to a CSV file."""
    FIELDNAMES = [
        "step",
        "loss",
        "val_loss",
        "val_ppl",
        "learning_rate",
        "tokens_seen",
        "tokens_per_sec",
        "memory_mb",
        "elapsed_time"
    ]

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

        # Initialize header
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def log(
        self,
        step: int,
        loss: float,
        val_loss: Optional[float] = None,
        val_ppl: Optional[float] = None,
        learning_rate: float = 0.0,
        tokens_seen: int = 0,
        tokens_per_sec: float = 0.0,
        memory_mb: float = 0.0,
        elapsed_time: float = 0.0
    ) -> None:
        """Appends a new metric record to CSV and in-memory history."""
        record = {
            "step": int(step),
            "loss": round(float(loss), 4),
            "val_loss": round(float(val_loss), 4) if val_loss is not None else "",
            "val_ppl": round(float(val_ppl), 4) if val_ppl is not None else "",
            "learning_rate": float(f"{learning_rate:.6e}"),
            "tokens_seen": int(tokens_seen),
            "tokens_per_sec": round(float(tokens_per_sec), 2),
            "memory_mb": round(float(memory_mb), 2),
            "elapsed_time": round(float(elapsed_time), 2)
        }
        self.history.append(record)

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Loads and returns logged history as pandas DataFrame."""
        return pd.read_csv(self.filepath)
