"""
Coding Challenge 4: Vocabulary Builder
Constructs token-to-integer mapping with special tokens <PAD>=0 and <UNK>=1.
"""

from typing import Dict, List


def build_vocabulary(sentences: List[str]) -> Dict[str, int]:
    """Build vocabulary from sentence strings.
    
    Args:
        sentences: List of text sentences.
        
    Returns:
        Dictionary mapping token to integer index.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for sentence in sentences:
        words = sentence.strip().lower().split()
        for w in words:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


if __name__ == "__main__":
    corpus = [
        "win free prize",
        "free cash now"
    ]
    vocab = build_vocabulary(corpus)
    print("Challenge 4: Vocabulary Builder")
    print(f"  Vocabulary: {vocab}")
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1
    assert "win" in vocab and "free" in vocab and "cash" in vocab
    print("  [SUCCESS] Vocabulary builder verified!")
