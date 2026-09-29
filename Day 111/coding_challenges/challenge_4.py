"""
Challenge 4: Implement WordPiece subword tokenization from scratch.
Greedy longest-match prefix search with ## continuation token handling.
"""
from typing import List, Set


class WordPieceTokenizer:
    def __init__(self, vocab: Set[str], unk_token: str = "[UNK]"):
        self.vocab = vocab
        self.unk_token = unk_token

    def tokenize_word(self, word: str) -> List[str]:
        sub_tokens = []
        start = 0
        is_bad = False

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


if __name__ == "__main__":
    vocab = {"un", "##believ", "##able", "play", "##ing", "[UNK]"}
    tok = WordPieceTokenizer(vocab)
    tokens_unbelievable = tok.tokenize_word("unbelievable")
    tokens_playing = tok.tokenize_word("playing")
    tokens_unknown = tok.tokenize_word("superman")

    print("unbelievable:", tokens_unbelievable)
    print("playing:     ", tokens_playing)
    print("unknown:     ", tokens_unknown)

    assert tokens_unbelievable == ["un", "##believ", "##able"]
    assert tokens_playing == ["play", "##ing"]
    assert tokens_unknown == ["[UNK]"]
    print("Challenge 4 passed successfully!")
