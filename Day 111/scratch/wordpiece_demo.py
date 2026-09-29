"""
WordPiece subword tokenization demonstration from scratch.
Implements the greedy longest-match first subword splitting algorithm.
"""
from typing import List, Set


class WordPieceTokenizerDemo:
    """
    Demonstration of WordPiece tokenization logic.
    For each word:
      1. Find longest matching prefix in vocabulary.
      2. For remaining characters, search for matching subwords prefixed with '##'.
      3. If any substring cannot be matched, emit [UNK].
    """
    def __init__(self, vocab: Set[str], unk_token: str = "[UNK]", max_input_chars_per_word: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize_word(self, word: str) -> List[str]:
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]

        is_bad = False
        start = 0
        sub_tokens = []

        while start < len(word):
            end = len(word)
            cur_substr = None

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr

                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1

            if cur_substr is None:
                is_bad = True
                break

            sub_tokens.append(cur_substr)
            start = end

        if is_bad:
            return [self.unk_token]
        return sub_tokens

    def tokenize(self, text: str) -> List[str]:
        words = text.lower().strip().split()
        tokens = []
        for w in words:
            tokens.extend(self.tokenize_word(w))
        return tokens


if __name__ == "__main__":
    demo_vocab = {
        "play", "##ing", "##ed", "##er", "##ful",
        "un", "##believ", "##able",
        "deep", "learn", "##ing",
        "cat", "dog", "[UNK]"
    }
    tokenizer = WordPieceTokenizerDemo(demo_vocab)
    for word in ["playing", "played", "playful", "unbelievable", "deeplearning", "spaceship"]:
        tokens = tokenizer.tokenize_word(word)
        print(f"'{word}' -> {tokens}")

    assert tokenizer.tokenize_word("playing") == ["play", "##ing"]
    assert tokenizer.tokenize_word("unbelievable") == ["un", "##believ", "##able"]
    assert tokenizer.tokenize_word("spaceship") == ["[UNK]"]
    print("WordPiece demo verification successful!")
