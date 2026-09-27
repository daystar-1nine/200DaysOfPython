from typing import List
from .vocabulary import Vocabulary

def encode_tokens(tokens: List[str], vocab: Vocabulary) -> List[int]:
    """
    Encodes list of string tokens to integer indices using Vocabulary.
    Unknown words are mapped to UNK_IDX.
    """
    return [vocab.get_idx(tok) for tok in tokens]

def encode_texts(texts: List[str], vocab: Vocabulary, tokenizer_fn) -> List[List[int]]:
    """
    Tokenizes and encodes a batch of raw texts.
    """
    encoded = []
    for text in texts:
        tokens = tokenizer_fn(text)
        encoded.append(encode_tokens(tokens, vocab))
    return encoded
