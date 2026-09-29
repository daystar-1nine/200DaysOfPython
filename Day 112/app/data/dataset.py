"""
Dataset loading and text partition routines for causal language modeling.
"""
from pathlib import Path
from typing import Tuple, Union
import torch
from app.tokenizer.char_tokenizer import CharacterTokenizer


def load_corpus(filepath: Union[str, Path]) -> str:
    """Reads and returns text file content."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_data_tensors(
    text: str,
    tokenizer: CharacterTokenizer,
    val_ratio: float = 0.10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes raw text into integer IDs and splits into train and validation tensors.
    """
    encoded_ids = tokenizer.encode(text)
    data = torch.tensor(encoded_ids, dtype=torch.long)

    n = len(data)
    split_idx = int(n * (1.0 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data
