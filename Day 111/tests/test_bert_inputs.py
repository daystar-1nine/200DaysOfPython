"""
Unit tests for BERT additive input embeddings and TensorDataset construction.
"""
import pytest
import numpy as np
import torch
from scratch.bert_input_representation import (
    BertInputEmbeddingsScratch,
    create_bert_input_embeddings
)
from app.models.embeddings import BertEmbeddings
from app.preprocessing.inputs import build_tensor_dataset


class TestBertInputEmbeddingsScratch:
    @pytest.fixture
    def scratch_embedder(self):
        return BertInputEmbeddingsScratch(
            vocab_size=1000,
            max_positions=128,
            type_vocab_size=2,
            hidden_dim=32,
            seed=42
        )

    def test_scratch_embedding_output_shape(self, scratch_embedder):
        batch_size, seq_len = 2, 8
        token_ids = np.random.randint(0, 1000, (batch_size, seq_len))
        position_ids = np.tile(np.arange(seq_len), (batch_size, 1))
        segment_ids = np.zeros((batch_size, seq_len), dtype=int)

        out = scratch_embedder.forward(token_ids, position_ids, segment_ids)
        assert out.shape == (batch_size, seq_len, 32)

    def test_additive_property_verification(self, scratch_embedder):
        token_ids = np.array([[10, 20]])
        pos_ids = np.array([[0, 1]])
        seg_ids = np.array([[0, 0]])

        total = scratch_embedder.forward(input_ids=token_ids, position_ids=pos_ids, token_type_ids=seg_ids)
        tok_vec = scratch_embedder.token_table[token_ids]
        pos_vec = scratch_embedder.position_table[pos_ids]
        seg_vec = scratch_embedder.segment_table[seg_ids]

        expected = tok_vec + pos_vec + seg_vec
        np.testing.assert_allclose(total, expected, atol=1e-6)

    def test_create_bert_input_embeddings_helper(self):
        tok_emb = np.ones((2, 4, 16))
        pos_emb = np.ones((2, 4, 16)) * 2
        seg_emb = np.ones((2, 4, 16)) * 3

        combined = create_bert_input_embeddings(tok_emb, pos_emb, seg_emb)
        expected = np.ones((2, 4, 16)) * 6
        np.testing.assert_allclose(combined, expected)


class TestBertEmbeddingsPyTorch:
    @pytest.fixture
    def pt_embedder(self):
        return BertEmbeddings(
            vocab_size=500,
            hidden_size=64,
            max_position_embeddings=128,
            type_vocab_size=2,
            dropout_prob=0.0
        )

    def test_pytorch_forward_shape(self, pt_embedder):
        input_ids = torch.randint(0, 500, (3, 16))
        out = pt_embedder(input_ids)
        assert out.shape == (3, 16, 64)

    def test_automatic_position_ids_generation(self, pt_embedder):
        input_ids = torch.randint(0, 500, (2, 8))
        out_auto = pt_embedder(input_ids, position_ids=None)
        manual_pos = torch.arange(8).unsqueeze(0).expand(2, -1)
        out_manual = pt_embedder(input_ids, position_ids=manual_pos)
        torch.testing.assert_close(out_auto, out_manual)

    def test_automatic_token_types_default_to_zero(self, pt_embedder):
        input_ids = torch.randint(0, 500, (2, 8))
        out_auto = pt_embedder(input_ids, token_type_ids=None)
        manual_types = torch.zeros_like(input_ids)
        out_manual = pt_embedder(input_ids, token_type_ids=manual_types)
        torch.testing.assert_close(out_auto, out_manual)

    def test_backprop_through_all_embedding_tables(self, pt_embedder):
        input_ids = torch.randint(0, 500, (2, 8))
        token_type_ids = torch.ones_like(input_ids)
        out = pt_embedder(input_ids, token_type_ids=token_type_ids)
        loss = out.sum()
        loss.backward()

        assert pt_embedder.word_embeddings.weight.grad is not None
        assert pt_embedder.position_embeddings.weight.grad is not None
        assert pt_embedder.token_type_embeddings.weight.grad is not None


class TestTensorDatasetBuilder:
    def test_dataset_shapes_and_types(self, tokenizer):
        texts = ["Message one", "Second spam message", "Third benign SMS"]
        labels = [0, 1, 0]
        max_len = 24

        ds = build_tensor_dataset(texts, labels, tokenizer, max_length=max_len)
        assert len(ds) == 3

        sample = ds[0]
        assert len(sample) == 4
        input_ids, mask, token_types, y = sample

        assert input_ids.shape == (max_len,)
        assert mask.shape == (max_len,)
        assert token_types.shape == (max_len,)
        assert y.shape == (1,)

        assert input_ids.dtype == torch.int64
        assert mask.dtype == torch.int64
        assert token_types.dtype == torch.int64
        assert y.dtype == torch.float32

    def test_labels_correctly_mapped(self, tokenizer):
        texts = ["Text A", "Text B"]
        labels = [0, 1]
        ds = build_tensor_dataset(texts, labels, tokenizer, max_length=16)

        assert ds[0][3].item() == 0.0
        assert ds[1][3].item() == 1.0
