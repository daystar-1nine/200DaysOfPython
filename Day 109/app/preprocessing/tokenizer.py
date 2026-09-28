import re
from typing import List

def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[\$£€]', ' currency ', text)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens
