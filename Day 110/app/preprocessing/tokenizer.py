"""
Tokenization module for SMS text processing.
"""
import re
from typing import List


def tokenize_text(text: str) -> List[str]:
    """
    Cleans and tokenizes raw string into lowercase word/number tokens.
    Retains currency indicators, digits, and alphanumeric tokens.
    """
    text = str(text).lower()
    # Normalize currency and punctuation symbols to retain spam signals
    text = re.sub(r"\$", " $ ", text)
    text = re.sub(r"£", " £ ", text)
    tokens = re.findall(r"[a-z0-9$£]+", text)
    return tokens
