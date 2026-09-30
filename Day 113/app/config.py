"""
Configuration management and architectural specifications for the Day 113 Scaling Lab.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class ModelConfig:
    name: str = "Tiny"
    vocab_size: int = 64
    context_length: int = 64
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    tie_weights: bool = True
    precision: str = "fp32"  # fp32, bf16, fp16


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    warmup_steps: int = 20
    max_steps: int = 200
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 40
    eval_iters: int = 10
    seed: int = 42
    scheduler_type: str = "cosine"  # cosine, linear, constant


# Pre-defined scaling tiers for comparative experimentation
SCALING_MODELS = {
    "tiny": ModelConfig(
        name="Tiny",
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        context_length=64
    ),
    "small": ModelConfig(
        name="Small",
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        context_length=64
    ),
    "medium": ModelConfig(
        name="Medium",
        embed_dim=384,
        num_heads=8,
        num_layers=8,
        context_length=64
    )
}


def load_config_from_yaml(filepath: Path) -> Dict[str, Any]:
    """Loads configuration dictionary from YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config_to_yaml(config_dict: Dict[str, Any], filepath: Path) -> None:
    """Serializes configuration dictionary to YAML."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
