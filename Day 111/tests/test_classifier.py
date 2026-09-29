"""
Unit tests for BertClassificationHead, BertSpamClassifier, freezing logic, and Trainer.
"""
import pytest
import torch
import torch.nn as nn
from app.models.classification_head import BertClassificationHead
from app.models.bert_classifier import BertSpamClassifier
from app.training.trainer import BertTrainer


class TestClassificationHead:
    def test_forward_output_shape(self):
        head = BertClassificationHead(hidden_size=128, num_classes=1)
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 1)

    def test_output_probabilities_bounded_zero_one(self):
        head = BertClassificationHead(hidden_size=128, num_classes=1)
        x = torch.randn(10, 128) * 5.0
        out = head(x)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_backprop_gradients_computed(self):
        head = BertClassificationHead(hidden_size=64, num_classes=1)
        x = torch.randn(2, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert head.dense.weight.grad is not None
        assert head.out_proj.weight.grad is not None


class TestBertSpamClassifier:
    @pytest.fixture(scope="class")
    def model(self):
        return BertSpamClassifier(model_name="prajjwal1/bert-tiny", freeze_bert=True)

    def test_frozen_parameter_counts(self, model):
        counts = model.count_parameters()
        assert counts["total"] > 4_000_000
        assert counts["trainable"] == 8_321
        assert counts["frozen"] == counts["total"] - 8_321

    def test_unfreeze_backbone(self, model):
        model.freeze_backbone(False)
        counts = model.count_parameters()
        assert counts["trainable"] == counts["total"]
        assert counts["frozen"] == 0

        # Re-freeze for next tests
        model.freeze_backbone(True)
        counts_refrozen = model.count_parameters()
        assert counts_refrozen["frozen"] > 0

    def test_forward_pass_outputs(self, model):
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(100, 2000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        probs, cls_repr, attentions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False
        )

        assert probs.shape == (batch_size, 1)
        assert cls_repr.shape == (batch_size, 128)
        assert attentions is None
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    def test_forward_with_attentions(self, model):
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(100, 2000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        probs, cls_repr, attentions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        assert attentions is not None
        assert len(attentions) == 2  # bert-tiny has 2 layers
        # Each layer: (batch_size, num_heads, seq_len, seq_len)
        assert attentions[0].shape == (batch_size, 2, seq_len, seq_len)


class TestTrainerHarness:
    def test_train_epoch_returns_metrics(self, dummy_dataloader):
        model = BertSpamClassifier(freeze_bert=True)
        trainer = BertTrainer(model=model, lr=1e-3)
        res = trainer.train_epoch(dummy_dataloader)

        assert "loss" in res
        assert "accuracy" in res
        assert "time_seconds" in res
        assert res["loss"] > 0.0
        assert 0.0 <= res["accuracy"] <= 1.0

    def test_evaluate_returns_full_evaluation(self, dummy_dataloader):
        model = BertSpamClassifier(freeze_bert=True)
        trainer = BertTrainer(model=model, lr=1e-3)
        eval_res = trainer.evaluate(dummy_dataloader)

        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "loss", "y_true", "y_pred", "y_probs"]:
            assert key in eval_res

    def test_extract_representations_dimensions(self, dummy_dataloader):
        model = BertSpamClassifier(freeze_bert=True)
        trainer = BertTrainer(model=model, lr=1e-3)
        cls_embs, probs, labels = trainer.extract_representations(dummy_dataloader)

        assert cls_embs.shape == (12, 128)
        assert probs.shape == (12,)
        assert labels.shape == (12,)
