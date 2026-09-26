import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from app.config import PAD_TOKEN, OOV_TOKEN, PAD_ID, OOV_ID, RANDOM_SEED

class TextPreprocessor:
    def __init__(self, max_vocab_size: int = 10000, max_len: int = 40):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_ID, OOV_TOKEN: OOV_ID}
        self.idx2word: Dict[int, str] = {PAD_ID: PAD_TOKEN, OOV_ID: OOV_TOKEN}
        self.vocab_size = 2 # Initial size with PAD and OOV
        self.is_fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple lowercase and non-alphanumeric split
        text = str(text).lower()
        # Keep alphabetic words only
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
        
    def fit(self, texts: List[str]):
        """
        Build vocabulary exclusively from texts provided (zero data leakage).
        """
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
            
        # Get most common words up to max_vocab_size - 2 (for PAD and OOV)
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.vocab_size = len(self.word2idx)
        self.is_fitted = True
        return self
        
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming.")
            
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            seq = [self.word2idx.get(token, OOV_ID) for token in tokens]
            sequences.append(seq)
            
        return self.pad_sequences(sequences)
        
    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        padded = np.full((len(sequences), self.max_len), PAD_ID, dtype=np.int32)
        for i, seq in enumerate(sequences):
            # Truncate if longer than max_len
            if len(seq) > self.max_len:
                padded[i, :] = seq[:self.max_len]
            # Pad if shorter (pre-padding is common in RNNs, but we'll use post-padding as standard, or pre-padding based on preference. Let's use post-padding for simplicity, but PyTorch pack_padded_sequence works well either way. We'll use post padding.)
            elif len(seq) > 0:
                padded[i, :len(seq)] = seq
        return padded

def create_stratified_splits(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into 70% Train, 15% Validation, 15% Test
    using stratified splitting to maintain class balance.
    """
    X = df[text_col]
    y = df[label_col]
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    
    # Second split: split the 30% temp into 15% val and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    train_df = pd.DataFrame({text_col: X_train, label_col: y_train})
    val_df = pd.DataFrame({text_col: X_val, label_col: y_val})
    test_df = pd.DataFrame({text_col: X_test, label_col: y_test})
    
    return train_df, val_df, test_df
