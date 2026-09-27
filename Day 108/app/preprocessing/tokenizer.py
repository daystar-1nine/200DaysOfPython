import re
from typing import List

def tokenize_text(text: str) -> List[str]:
    """
    Standard lowercased alphanumeric tokenizer for SMS text sequences.
    Preserves currency and exclamation signals.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Lowercase
    text = text.lower()
    # Normalize special tokens like currency symbols
    text = re.sub(r'[\$£€]', ' currency ', text)
    # Split words and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens
