"""
Standalone Character Tokenizer scratch implementation.
"""
from typing import List, Dict, Union


class CharacterTokenizerScratch:
    def __init__(self, chars: List[str]):
        self.vocab = sorted(list(set(chars)))
        self.char2idx: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char: Dict[int, str] = {i: ch for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, ids: Union[List[int], List[float]]) -> str:
        return "".join([self.idx2char[int(i)] for i in ids if int(i) in self.idx2char])


if __name__ == "__main__":
    sample = "hello world"
    tok = CharacterTokenizerScratch(list(sample))
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print("Sample: ", sample)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    assert decoded == sample
    assert tok.vocab_size == len(set(sample))
    print("Character Tokenizer Scratch verified!")
