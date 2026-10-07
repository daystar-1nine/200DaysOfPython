"""
Configuration dataclasses for Day 115: Preference Optimization & RLHF.
Defines parameters for Reward Modeling, Direct Preference Optimization (DPO),
Model Architectures, and Inference Generation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class ModelConfig:
    """Architectural configuration for MiniGPT backbone."""
    name: str = "MiniGPT"
    vocab_size: int = 104        # Matches character vocabulary + special tokens
    context_length: int = 128    # Maximum context window
    embed_dim: int = 128         # Hidden embedding dimension
    num_heads: int = 4           # Multi-head attention heads
    num_layers: int = 4          # Number of Transformer blocks
    dropout: float = 0.1         # Dropout probability
    tie_weights: bool = True     # Tie token embeddings with LM head
    # Special tokens
    pad_token: str = "<|pad|>"
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"


@dataclass
class RewardModelConfig:
    """Configuration for Reward Model architecture."""
    name: str = "MiniGPT-Reward"
    pooling_method: str = "last"  # "last", "mean", or "cls"
    dropout: float = 0.1
    head_hidden_dim: Optional[int] = None  # None for single linear projection, int for 2-layer MLP


@dataclass
class RewardTrainingConfig:
    """Hyperparameters for Reward Model training."""
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 10
    grad_clip: float = 1.0
    eval_interval: int = 15
    seed: int = 42
    margin: float = 0.0          # Bradley-Terry margin: -log sigma(r_c - r_r - margin)


@dataclass
class DPOTrainingConfig:
    """Hyperparameters for Direct Preference Optimization (DPO)."""
    beta: float = 0.1            # Temperature parameter controlling deviation from reference model
    learning_rate: float = 5e-5
    min_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 10
    grad_clip: float = 1.0
    eval_interval: int = 15
    seed: int = 42
    reference_model_path: Optional[str] = None
    loss_type: str = "sigmoid"   # "sigmoid" (standard DPO), "hinge", or "ipo"


@dataclass
class GenerationConfig:
    """Decoding hyperparameters for autoregressive inference."""
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.2


def load_yaml_config(filepath: Path) -> Dict[str, Any]:
    """Loads configuration dictionary from YAML."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_config(config_dict: Dict[str, Any], filepath: Path) -> None:
    """Saves configuration dictionary to YAML."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
