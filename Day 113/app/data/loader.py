"""
Raw text loader and document chunking routines for LLM pretraining pipelines.
"""
from pathlib import Path
from typing import List, Dict, Union, Tuple


def load_raw_text(filepath: Union[str, Path]) -> str:
    """Reads and returns text file content."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset corpus file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_documents(text: str, delimiter: str = "\n\n") -> List[str]:
    """
    Splits continuous corpus text into discrete document chunks based on delimiter.
    Discards empty chunks.
    """
    raw_chunks = text.split(delimiter)
    docs = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return docs


def build_character_vocab(text: str, unk_token: str = "<unk>") -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Extracts unique characters from corpus and constructs deterministic bidirectional mapping.
    """
    clean_chars = sorted(list(set(text) - {unk_token}))
    vocab = [unk_token] + clean_chars
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for idx, ch in enumerate(vocab)}
    return vocab, char2idx, idx2char
