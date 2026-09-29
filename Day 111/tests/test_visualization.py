"""
Unit tests for visualization functions (attention heatmaps, PCA projections, and benchmark charts).
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from app.models.bert_classifier import BertSpamClassifier
from app.visualization.attention import visualize_bert_attention
from app.visualization.embeddings import visualize_cls_embeddings
from app.visualization.benchmark import (
    plot_benchmark_comparison,
    plot_training_curves,
    plot_threshold_curve
)


class TestVisualizationModules:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
        model = BertSpamClassifier(model_name="prajjwal1/bert-tiny", freeze_bert=True)
        return model, tokenizer

    def test_visualize_bert_attention_creates_file(self, tmp_path, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        out_img = tmp_path / "attn.png"
        visualize_bert_attention("Hello test world", model, tokenizer, out_img, layer_idx=-1)
        assert out_img.exists()
        assert out_img.stat().st_size > 1000

    def test_visualize_cls_embeddings_creates_file(self, tmp_path):
        cls_embs = np.random.randn(20, 128)
        labels = np.array([0] * 12 + [1] * 8)
        out_img = tmp_path / "cls_pca.png"
        visualize_cls_embeddings(cls_embs, labels, out_img)
        assert out_img.exists()
        assert out_img.stat().st_size > 1000

    def test_plot_benchmark_comparison_creates_file(self, tmp_path):
        df = pd.DataFrame({
            "Model": ["TF-IDF", "BERT"],
            "Accuracy": [0.95, 0.98],
            "Precision": [0.92, 0.96],
            "Recall": [0.90, 0.95],
            "F1": [0.91, 0.955],
            "ROC-AUC": [0.98, 0.99]
        })
        out_img = tmp_path / "bench.png"
        plot_benchmark_comparison(df, out_img)
        assert out_img.exists()
        assert out_img.stat().st_size > 1000

    def test_plot_training_curves_creates_file(self, tmp_path):
        history_a = {"epoch": [1, 2], "train_loss": [0.7, 0.5], "val_loss": [0.65, 0.48], "val_f1": [0.8, 0.85]}
        history_b = {"epoch": [1, 2], "train_loss": [0.68, 0.4], "val_loss": [0.60, 0.38], "val_f1": [0.82, 0.91]}
        out_img = tmp_path / "curves.png"
        plot_training_curves(history_a, history_b, out_img)
        assert out_img.exists()
        assert out_img.stat().st_size > 1000

    def test_plot_threshold_curve_creates_file(self, tmp_path):
        thresh_df = pd.DataFrame({
            "threshold": [0.2, 0.5, 0.8],
            "precision": [0.85, 0.92, 0.98],
            "recall": [0.98, 0.95, 0.80],
            "f1": [0.91, 0.935, 0.88]
        })
        out_img = tmp_path / "thresh.png"
        plot_threshold_curve(thresh_df, out_img)
        assert out_img.exists()
        assert out_img.stat().st_size > 1000
