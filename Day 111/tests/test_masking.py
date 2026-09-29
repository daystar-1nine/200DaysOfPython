"""
Unit tests for Masked Language Modeling (80/10/10 rule) and Next Sentence Prediction pair creation.
"""
import numpy as np
import pytest
from scratch.masked_language_modeling import mask_tokens_mlm
from app.preprocessing.masking import mask_tokens
from scratch.next_sentence_prediction import create_nsp_pairs


class TestMaskedLanguageModeling:
    @pytest.fixture
    def token_sequence(self):
        # 101: [CLS], 102: [SEP], 103: [MASK]
        return [101, 2023, 2003, 1037, 3231, 2005, 1037, 2742, 102]

    def test_special_tokens_never_masked(self, token_sequence):
        for seed in range(20):
            masked_ids, labels = mask_tokens(
                token_sequence,
                mask_token_id=103,
                cls_token_id=101,
                sep_token_id=102,
                mask_prob=0.5,
                seed=seed
            )
            assert masked_ids[0] == 101
            assert masked_ids[-1] == 102
            assert labels[0] == -100
            assert labels[-1] == -100

    def test_labels_unmasked_are_minus_100(self, token_sequence):
        masked_ids, labels = mask_tokens(
            token_sequence,
            mask_token_id=103,
            mask_prob=0.3,
            seed=42
        )
        for original, masked, label in zip(token_sequence, masked_ids, labels):
            if label != -100:
                assert label == original
            else:
                assert masked == original

    def test_output_length_preserved(self, token_sequence):
        masked_ids, labels = mask_tokens(token_sequence, mask_token_id=103, seed=42)
        assert len(masked_ids) == len(token_sequence)
        assert len(labels) == len(token_sequence)

    def test_zero_masking_probability(self, token_sequence):
        masked_ids, labels = mask_tokens(token_sequence, mask_prob=0.0, seed=42)
        assert masked_ids == token_sequence
        assert all(lbl == -100 for lbl in labels)

    def test_deterministic_with_seed(self, token_sequence):
        m1, l1 = mask_tokens(token_sequence, seed=123)
        m2, l2 = mask_tokens(token_sequence, seed=123)
        assert m1 == m2
        assert l1 == l2

    def test_scratch_mlm_returns_expected_types(self, token_sequence):
        masked_arr, label_arr = mask_tokens_mlm(
            token_ids=np.array(token_sequence),
            mask_prob=0.3,
            mask_token_id=103,
            vocab_size=30522,
            seed=42
        )
        assert isinstance(masked_arr, np.ndarray)
        assert isinstance(label_arr, np.ndarray)
        assert masked_arr.shape == (len(token_sequence),)
        assert label_arr.shape == (len(token_sequence),)

    def test_eighty_percent_mask_token_substitution(self):
        # Large sequence to statistically verify the 80/10/10 distribution
        tokens = [101] + list(range(200, 1200)) + [102]
        masked_ids, labels = mask_tokens(tokens, mask_prob=0.5, seed=42)

        masked_count = 0
        mask_token_replacements = 0
        for orig, masked, lbl in zip(tokens, masked_ids, labels):
            if lbl != -100:
                masked_count += 1
                if masked == 103:
                    mask_token_replacements += 1

        # In 80% rule, approx 80% (+/- 10%) should be [MASK]
        ratio = mask_token_replacements / max(masked_count, 1)
        assert 0.65 <= ratio <= 0.95


class TestNextSentencePrediction:
    @pytest.fixture
    def corpus(self):
        return [
            "Transformers use self-attention to model long-range context.",
            "BERT is a bidirectional encoder trained with MLM.",
            "PyTorch makes deep learning research productive and flexible.",
            "Evaluating models on unseen test sets detects overfitting."
        ]

    def test_create_nsp_pairs_length(self, corpus):
        pairs = create_nsp_pairs(corpus, seed=42)
        assert len(pairs) == len(corpus) - 1

    def test_nsp_pair_structure(self, corpus):
        pairs = create_nsp_pairs(corpus, seed=42)
        for pair in pairs:
            assert "sentence_a" in pair
            assert "sentence_b" in pair
            assert "label" in pair
            assert pair["label"] in (0, 1)

    def test_nsp_label_distribution(self, corpus):
        # Generate with fixed seed and ensure both 0 (NotNext) and 1 (IsNext) occur
        extended_corpus = corpus * 5
        pairs = create_nsp_pairs(extended_corpus, seed=42)
        labels = [p["label"] for p in pairs]
        assert 0 in labels
        assert 1 in labels

    def test_nsp_consecutive_is_next(self, corpus):
        pairs = create_nsp_pairs(corpus, seed=1)
        for pair in pairs:
            if pair["label"] == 1:
                idx_a = corpus.index(pair["sentence_a"])
                idx_b = corpus.index(pair["sentence_b"])
                assert idx_b == idx_a + 1
