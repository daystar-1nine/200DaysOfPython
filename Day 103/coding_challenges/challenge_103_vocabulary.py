"""Challenge 2 & 3: Vocabulary and Integer Mapping."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from collections import Counter
from typing import List, Dict, Tuple

def build_vocabulary(corpus: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tokens = [w for doc in corpus for w in doc.lower().split()]
    vocab = sorted(list(set(tokens)))
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word

def run():
    print("=== CHALLENGE 103: VOCABULARY & INTEGER MAPPING ===")
    corpus = ["the cat drinks milk", "the dog drinks water"]
    w2id, id2w = build_vocabulary(corpus)
    print("word2id:", w2id)
    print("id2word:", id2w)
    assert len(w2id) == 6
    assert id2w[w2id["cat"]] == "cat"
    print("[SUCCESS] Vocabulary building verified!")

if __name__ == "__main__":
    run()
