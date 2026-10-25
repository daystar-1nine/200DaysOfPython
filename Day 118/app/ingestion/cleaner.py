"""
Text cleaning and normalization utilities for document ingestion and retrieval.
"""
import re
import unicodedata
from typing import List


def clean_text(text: str) -> str:
    """
    Cleans raw document text:
    - Normalizes Unicode characters to NFC
    - Replaces non-breaking spaces with standard space
    - Collapses 3 or more consecutive newlines into 2
    - Strips non-printable control characters (except tab and newline)
    - Trims surrounding whitespace
    """
    if not text:
        return ""
    
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    
    # Replace non-breaking spaces and carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    
    # Strip null bytes and non-printable control characters
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    
    # Collapse multiple consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Collapse horizontal spaces while keeping line breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def tokenize_words(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenizes text into words/tokens for length estimation, BM25, and search.
    Preserves alphanumeric terms, hyphens within words, and underscores.
    """
    if not text:
        return []
    # Match words, alphanumeric terms, and technical identifiers like SQLSTATE or 404
    tokens = re.findall(r"\b[\w\-]+\b", text)
    return [t.lower() for t in tokens] if lowercase else tokens


def extract_sentences(text: str) -> List[str]:
    """Splits text into constituent sentences preserving terminals."""
    if not text or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def remove_markdown_noise(text: str) -> str:
    """Strips markdown links and headers for cleaner semantic representation."""
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[*_~`]", "", text)
    return clean_text(text)


def normalize_for_search(text: str) -> str:
    """Canonical lowercasing and punctuation stripping for search indexing."""
    text = clean_text(text).lower()
    return re.sub(r"[^\w\s]", " ", text)
