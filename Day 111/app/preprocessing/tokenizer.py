"""
BERT Tokenizer wrapper and token inspection routines.
"""
from typing import List, Dict, Any, Union
import torch
from transformers import BertTokenizer


def get_bert_tokenizer(model_name: str = "prajjwal1/bert-tiny") -> BertTokenizer:
    """Loads and returns BertTokenizer for the specified checkpoint."""
    return BertTokenizer.from_pretrained(model_name)


def tokenize_texts(
    texts: List[str],
    tokenizer: BertTokenizer,
    max_length: int = 128,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a list of strings with uniform padding and truncation.
    Returns PyTorch tensors for:
      - input_ids
      - attention_mask
      - token_type_ids
    """
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors
    )


def inspect_tokenization(text: str, tokenizer: BertTokenizer) -> Dict[str, Any]:
    """
    Breaks down a text into WordPiece tokens, integer IDs, and special tokens.
    """
    encoding = tokenizer(
        text,
        padding=False,
        truncation=True,
        return_tensors=None
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    return {
        "text": text,
        "tokens": tokens,
        "input_ids": encoding["input_ids"],
        "ids": encoding["input_ids"],
        "attention_mask": encoding.get("attention_mask", [1] * len(tokens)),
        "token_type_ids": encoding.get("token_type_ids", [0] * len(tokens)),
        "num_tokens": len(tokens),
        "subword_count": len([t for t in tokens if t.startswith("##")])
    }
