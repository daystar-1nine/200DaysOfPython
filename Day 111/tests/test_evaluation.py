"""
Unit tests for classification metrics, threshold sweep, error analysis, and benchmarking.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from app.evaluation.metrics import compute_metrics
from app.evaluation.threshold import sweep_thresholds
from app.evaluation.errors import analyze_errors
from app.training.benchmarking import run_tfidf_baseline, build_benchmark_table


class TestMetricsCalculation:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_probs = [0.1, 0.2, 0.9, 0.85]
        m = compute_metrics(y_true, y_pred, y_probs)

        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["tp"] == 2 and m["tn"] == 2
        assert m["fp"] == 0 and m["fn"] == 0

    def test_completely_inverted_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        y_probs = [0.9, 0.8, 0.1, 0.2]
        m = compute_metrics(y_true, y_pred, y_probs)

        assert m["accuracy"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["tp"] == 0 and m["tn"] == 0
        assert m["fp"] == 2 and m["fn"] == 2

    def test_zero_division_safety(self):
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_probs = [0.1, 0.2, 0.1, 0.05]
        m = compute_metrics(y_true, y_pred, y_probs)

        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_confusion_matrix_sum_matches_sample_count(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        y_probs = [0.2, 0.4, 0.1, 0.8, 0.9, 0.6]
        m = compute_metrics(y_true, y_pred, y_probs)

        assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y_true)


class TestThresholdAnalysis:
    @pytest.fixture
    def synthetic_probs(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_probs = [0.05, 0.15, 0.35, 0.65, 0.45, 0.75, 0.85, 0.95]
        return y_true, y_probs

    def test_sweep_thresholds_columns(self, synthetic_probs):
        y_true, y_probs = synthetic_probs
        df = sweep_thresholds(y_true, y_probs, thresholds=[0.2, 0.5, 0.8])

        expected_cols = ["threshold", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
        for col in expected_cols:
            assert col in df.columns
        assert len(df) == 3

    def test_sweep_thresholds_extreme_boundaries(self, synthetic_probs):
        y_true, y_probs = synthetic_probs
        df = sweep_thresholds(y_true, y_probs, thresholds=[0.01, 0.99])

        # At threshold 0.01, all are predicted positive -> recall is 1.0
        assert df.loc[df["threshold"] == 0.01, "recall"].values[0] == 1.0

    def test_export_thresholds_csv(self, tmp_path, synthetic_probs):
        y_true, y_probs = synthetic_probs
        out_csv = tmp_path / "thresholds.csv"
        sweep_thresholds(y_true, y_probs, thresholds=[0.3, 0.5, 0.7], output_path=out_csv)
        assert out_csv.exists()
        loaded = pd.read_csv(out_csv)
        assert len(loaded) == 3


class TestErrorAnalysis:
    def test_detects_false_positive_and_false_negative(self):
        texts = ["Meeting at 5", "Win free prize", "Call urgent", "Normal update"]
        y_true = [0, 1, 0, 0]
        y_pred = [0, 0, 1, 0]  # sample 1 is FN, sample 2 is FP
        y_probs = [0.1, 0.3, 0.85, 0.05]

        errors = analyze_errors(texts, y_true, y_pred, y_probs)
        assert len(errors) == 2

        err_types = set(errors["error_type"])
        assert "False Positive" in err_types
        assert "False Negative" in err_types

    def test_no_errors_when_perfect(self):
        texts = ["Text A", "Text B"]
        y_true = [0, 1]
        y_pred = [0, 1]
        y_probs = [0.1, 0.9]

        errors = analyze_errors(texts, y_true, y_pred, y_probs)
        assert len(errors) == 0

    def test_errors_sorted_by_confidence(self):
        texts = ["Text 1", "Text 2"]
        y_true = [0, 0]
        y_pred = [1, 1]
        y_probs = [0.60, 0.95]

        errors = analyze_errors(texts, y_true, y_pred, y_probs)
        assert errors.iloc[0]["error_confidence"] >= errors.iloc[1]["error_confidence"]

    def test_errors_export_csv(self, tmp_path):
        texts = ["Spam promo"]
        y_true = [1]
        y_pred = [0]
        y_probs = [0.2]
        out_csv = tmp_path / "errors.csv"

        analyze_errors(texts, y_true, y_pred, y_probs, output_path=out_csv)
        assert out_csv.exists()


class TestBenchmarkBuilding:
    def test_run_tfidf_baseline(self):
        train_texts = ["hello friend", "win prize cash claim", "meeting tomorrow", "claim urgent gift"]
        train_labels = [0, 1, 0, 1]
        test_texts = ["hello how are you", "win cash prize"]
        test_labels = [0, 1]

        res = run_tfidf_baseline(train_texts, train_labels, test_texts, test_labels)
        assert "accuracy" in res
        assert "f1" in res
        assert "parameters" in res
        assert "training_time_seconds" in res

    def test_build_benchmark_table_structure(self, tmp_path):
        dummy_metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.90, "f1": 0.91, "roc_auc": 0.98}
        out_csv = tmp_path / "bench.csv"

        df = build_benchmark_table(dummy_metrics, dummy_metrics, dummy_metrics, output_path=out_csv)
        assert len(df) == 5
        assert "Model" in df.columns
        assert "Parameters" in df.columns
        assert "F1" in df.columns
        assert out_csv.exists()
