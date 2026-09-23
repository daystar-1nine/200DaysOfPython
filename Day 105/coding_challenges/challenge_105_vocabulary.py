"""
Coding Challenges 105.1 - 105.3: Vocabulary Construction, Token Encoding, and Decoding.
Day 105: Neural NLP & Text Classification.
"""

from collections import Counter
from typing import Dict, List


class Challenge105Vocabulary:
    """Builds token-to-integer mappings with <PAD>=0 and <UNK>=1."""

    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: Dict[int, str] = {0: pad_token, 1: unk_token}

    def fit(self, texts: List[str], min_freq: int = 1) -> "Challenge105Vocabulary":
        """Challenge 1: Build vocabulary from list of text sentences."""
        counts = Counter(w.lower() for t in texts for w in t.split())
        words = sorted([w for w, c in counts.items() if c >= min_freq])

        next_id = 2
        for w in words:
            if w not in self.word2idx:
                self.word2idx[w] = next_id
                self.idx2word[next_id] = w
                next_id += 1
        return self

    def encode(self, text: str) -> List[int]:
        """Challenge 2: Encode raw text into integer token IDs."""
        tokens = text.lower().split()
        return [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens]

    def decode(self, ids: List[int], skip_special: bool = False) -> List[str]:
        """Challenge 3: Decode integer IDs back to word tokens."""
        words = []
        for i in ids:
            w = self.idx2word.get(i, self.unk_token)
            if skip_special and w in (self.pad_token, self.unk_token):
                continue
            words.append(w)
        return words


if __name__ == "__main__":
    corpus = [
        "Python is awesome",
        "Machine learning with Python",
        "Deep neural networks"
    ]

    vocab = Challenge105Vocabulary().fit(corpus)
    print("Vocab size:", len(vocab.word2idx))

    # Test Challenge 2: Encoding
    sample = "Python deep learning robot"
    encoded = vocab.encode(sample)
    print(f"Encoded '{sample}':", encoded)
    assert encoded[0] == vocab.word2idx["python"]
    assert encoded[-1] == 1  # 'robot' is unknown -> <UNK>=1

    # Test Challenge 3: Decoding
    decoded = vocab.decode(encoded)
    print("Decoded:", decoded)
    assert decoded[0] == "python"
    assert decoded[-1] == "<UNK>"

    print("[SUCCESS] Challenges 105.1 - 105.3 Passed!")
