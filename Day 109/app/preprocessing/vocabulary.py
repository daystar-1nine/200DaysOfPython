from typing import List, Dict, Iterable
from collections import Counter
from app.config import PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX, MAX_VOCAB_SIZE

class Vocabulary:
    def __init__(self, max_size: int = MAX_VOCAB_SIZE, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_counts: Counter = Counter()
        self.is_fitted: bool = False

    def fit(self, tokenized_texts: Iterable[List[str]]) -> "Vocabulary":
        self.word_counts = Counter()
        for tokens in tokenized_texts:
            self.word_counts.update(tokens)
            
        eligible_words = [
            word for word, count in self.word_counts.most_common()
            if count >= self.min_freq and word not in (PAD_TOKEN, UNK_TOKEN)
        ]
        
        limit = self.max_size - 2
        top_words = eligible_words[:limit]
        
        for idx, word in enumerate(top_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.is_fitted = True
        return self

    def __len__(self) -> int:
        return len(self.word2idx)

    def get_idx(self, word: str) -> int:
        return self.word2idx.get(word, UNK_IDX)

    def get_word(self, idx: int) -> str:
        return self.idx2word.get(idx, UNK_TOKEN)
