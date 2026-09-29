"""
Unit tests for scratch algorithmic implementations (WordPiece, Additive Embeddings, MLM, NSP).
"""
import numpy as np
import pytest
from scratch.wordpiece_demo import WordPieceTokenizerScratch
from scratch.bert_input_representation import BertInputEmbeddingsScratch
from scratch.masked_language_modeling import mask_tokens_mlm
from scratch.next_sentence_prediction import create_nsp_pairs


class TestScratchWordPieceAlgorithm:
    def test_longest_match_first(self):
        # "pretraining": if "pre" and "train" and "##ing" exist, it should greedily pick longest matches
        vocab = ["pre", "train", "##train", "##ing", "[UNK]"]
        tok = WordPieceTokenizerScratch(vocab=vocab)
        tokens = tok.tokenize("pretraining")
        assert tokens == ["pre", "##train", "##ing"]

    def test_single_character_fallback(self):
        vocab = ["a", "##b", "##c", "##d", "[UNK]"]
        tok = WordPieceTokenizerScratch(vocab=vocab)
        tokens = tok.tokenize("abcd")
        assert tokens == ["a", "##b", "##c", "##d"]

    def test_oov_character_produces_unk(self):
        vocab = ["a", "b", "##c", "[UNK]"]
        tok = WordPieceTokenizerScratch(vocab=vocab)
        tokens = tok.tokenize("abz")
        assert tokens == ["[UNK]"]


class TestScratchAdditiveEmbeddings:
    def test_position_embeddings_distinct_per_position(self):
        embedder = BertInputEmbeddingsScratch(vocab_size=100, max_positions=10, type_vocab_size=2, hidden_dim=16, seed=1)
        pos0 = embedder.position_table[0]
        pos1 = embedder.position_table[1]
        assert not np.allclose(pos0, pos1)

    def test_segment_embeddings_distinct_for_pair(self):
        embedder = BertInputEmbeddingsScratch(vocab_size=100, max_positions=10, type_vocab_size=2, hidden_dim=16, seed=1)
        seg0 = embedder.segment_table[0]
        seg1 = embedder.segment_table[1]
        assert not np.allclose(seg0, seg1)

    def test_zero_vector_addition(self):
        embedder = BertInputEmbeddingsScratch(vocab_size=10, max_positions=10, type_vocab_size=2, hidden_dim=8)
        embedder.position_table.fill(0.0)
        embedder.segment_table.fill(0.0)

        token_ids = np.array([[2, 5]])
        pos_ids = np.array([[0, 1]])
        seg_ids = np.array([[0, 0]])

        result = embedder.forward(token_ids, pos_ids, seg_ids)
        expected = embedder.token_table[token_ids]
        np.testing.assert_allclose(result, expected)


class TestScratchMLMAndNSP:
    def test_mlm_random_replacement_within_vocab(self):
        seq = np.array([101, 10, 20, 30, 40, 50, 60, 70, 80, 90, 102])
        masked, labels = mask_tokens_mlm(seq, mask_prob=0.8, mask_token_id=103, vocab_size=500, seed=42)

        for token in masked:
            assert 0 <= token < 500

    def test_mlm_no_mutation_of_input(self):
        original = np.array([101, 55, 66, 77, 102])
        copy = original.copy()
        mask_tokens_mlm(original, mask_prob=0.5, seed=42)
        np.testing.assert_array_equal(original, copy)

    def test_nsp_deterministic_split(self):
        corpus = ["A sentence", "B sentence", "C sentence", "D sentence"]
        pairs1 = create_nsp_pairs(corpus, seed=99)
        pairs2 = create_nsp_pairs(corpus, seed=99)
        assert pairs1 == pairs2
