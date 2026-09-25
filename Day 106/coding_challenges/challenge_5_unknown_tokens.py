"""
Coding Challenge 5: Unknown Token Handling
Encodes a sentence into integer token IDs, mapping unknown words to <UNK>.
"""

from typing import Dict, List


def encode_sentence(sentence: str, vocabulary: Dict[str, int]) -> List[int]:
    """Encode text string into integer IDs using vocabulary.
    
    Args:
        sentence: Raw text string.
        vocabulary: Dictionary of token to int ID.
        
    Returns:
        List of integer IDs.
    """
    unk_id = vocabulary.get("<UNK>", 1)
    words = sentence.strip().lower().split()
    return [vocabulary.get(w, unk_id) for w in words]


if __name__ == "__main__":
    vocab = {"<PAD>": 0, "<UNK>": 1, "win": 2, "cash": 3}
    text = "win extra cash reward"
    encoded = encode_sentence(text, vocab)
    print("Challenge 5: Unknown Token Handling")
    print(f"  Input text : '{text}'")
    print(f"  Encoded IDs: {encoded}")
    assert encoded == [2, 1, 3, 1]
    print("  [SUCCESS] Unknown tokens mapped to 1!")
