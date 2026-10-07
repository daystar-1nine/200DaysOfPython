"""
Configuration dataclasses and paths for Day 116: LLM Evaluation Harness.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

DAY_116_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GenerationConfig:
    """Generation hyperparameters for evaluation runs."""
    temperature: float = 0.0          # 0.0 = greedy deterministic evaluation
    top_k: int = 40
    top_p: float = 0.9
    max_new_tokens: int = 64
    do_sample: bool = False           # False for deterministic reproducibility
    repetition_penalty: float = 1.1
    seed: int = 42


@dataclass
class EvalHarnessConfig:
    """Core configuration for the evaluation harness and benchmark pipeline."""
    # Data paths
    dataset_path: Path = DAY_116_ROOT / "data" / "evaluation.jsonl"
    reasoning_path: Path = DAY_116_ROOT / "data" / "reasoning.jsonl"
    coding_path: Path = DAY_116_ROOT / "data" / "coding.jsonl"
    safety_path: Path = DAY_116_ROOT / "data" / "safety.jsonl"
    factuality_path: Path = DAY_116_ROOT / "data" / "factuality.jsonl"

    # Output paths
    output_dir: Path = DAY_116_ROOT / "outputs"
    raw_dir: Path = DAY_116_ROOT / "outputs" / "raw"
    metrics_dir: Path = DAY_116_ROOT / "outputs" / "metrics"
    charts_dir: Path = DAY_116_ROOT / "outputs" / "charts"
    reports_dir: Path = DAY_116_ROOT / "outputs" / "reports"

    # Evaluation parameters
    models_to_evaluate: List[str] = field(default_factory=lambda: ["base", "sft", "dpo"])
    categories: List[str] = field(default_factory=lambda: [
        "python", "data_science", "machine_learning", "dbms", "mathematics",
        "general_knowledge", "reasoning", "coding", "instruction_following", "safety"
    ])
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    batch_size: int = 16
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_directories(self) -> None:
        """Creates all necessary output directories."""
        for d in [self.output_dir, self.raw_dir, self.metrics_dir, self.charts_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
