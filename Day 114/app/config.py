"""
Configuration management and hyperparameter specifications for Day 114 Instruction Fine-Tuning.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class ChatModelConfig:
    """Architectural configuration for MiniGPT-Chat."""
    name: str = "MiniGPT-Chat"
    vocab_size: int = 128  # Character-level / byte vocab including special tokens
    context_length: int = 128
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    tie_weights: bool = True
    precision: str = "fp32"
    # Special token strings
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"
    pad_token: str = "<|pad|>"


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA) fine-tuning."""
    enabled: bool = False
    r: int = 8                # Rank of low-rank update
    lora_alpha: float = 16.0  # Scaling factor alpha
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])


@dataclass
class SFTTrainingConfig:
    """Hyperparameters for Supervised Fine-Tuning (SFT)."""
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 15
    num_epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 20
    eval_iters: int = 10
    seed: int = 42
    scheduler_type: str = "cosine"  # cosine, linear, constant
    mask_prompt_loss: bool = True   # True: compute loss only on assistant tokens (-100 on prompt)


@dataclass
class GenerationConfig:
    """Decoding parameters for autoregressive text generation."""
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


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
