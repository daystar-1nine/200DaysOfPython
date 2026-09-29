"""
BERT input creation, batch tensor dataset building, and tokenization experiment generation.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from app.preprocessing.tokenizer import inspect_tokenization


def build_tensor_dataset(
    texts: List[str],
    labels: List[int],
    tokenizer: BertTokenizer,
    max_length: int = 128
) -> TensorDataset:
    """
    Constructs a PyTorch TensorDataset containing:
        (input_ids, attention_mask, token_type_ids, labels)
    """
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    token_type_ids = encoded.get(
        "token_type_ids",
        torch.zeros_like(encoded["input_ids"])
    )

    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        token_type_ids,
        y_tensor
    )


def generate_tokenization_examples_csv(
    tokenizer: BertTokenizer,
    output_path: Path,
    words: List[str] = None
) -> pd.DataFrame:
    """
    Analyzes subword decomposition for morphological variants and exports to CSV.
    """
    if words is None:
        words = ["playing", "played", "playful", "unbelievable", "congratulations", "disproportionate"]

    records = []
    for word in words:
        info = inspect_tokenization(word, tokenizer)
        # Exclude [CLS] and [SEP] to focus on the subword components
        subwords = info["tokens"][1:-1]
        records.append({
            "Text": word,
            "Tokens": " ".join(subwords),
            "Number of Tokens": len(subwords)
        })

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
