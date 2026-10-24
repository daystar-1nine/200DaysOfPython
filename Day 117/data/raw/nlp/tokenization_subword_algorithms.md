# Subword Tokenization Algorithms in Modern NLP

Tokenization converts raw character strings into numerical token IDs suitable for neural models. Classical word-level tokenization struggles with out-of-vocabulary (OOV) terms and morphological variations, while character-level tokenization produces excessively long sequence lengths that stress attention mechanisms.

Modern NLP relies on subword tokenization algorithms. Byte-Pair Encoding (BPE) starts with individual characters and iteratively merges the most frequent adjacent character pairs into new tokens. WordPiece selects merges that maximize language model likelihood. Unigram Tokenizer starts with a large candidate vocabulary and iteratively prunes tokens that cause the smallest increase in unigram loss. Byte-level BPE (used in GPT) operates directly on UTF-8 bytes to handle any Unicode string without unknown token artifacts.
