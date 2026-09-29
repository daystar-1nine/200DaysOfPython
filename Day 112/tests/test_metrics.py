"""
Unit tests for Day 112 evaluation metrics (Perplexity, Repetition, Distinct-N).
"""
import math
import pytest
from app.evaluation.metrics import (
    compute_perplexity,
    compute_distinct_n,
    evaluate_text_diversity
)
from app.evaluation.repetition import (
    repetition_rate,
    analyze_repetition
)


class TestPerplexityMetrics:
    def test_perplexity_zero_loss(self):
        # e^0 == 1.0 (ideal theoretical limit)
        ppl = compute_perplexity(0.0)
        assert math.isclose(ppl, 1.0, rel_tol=1e-4)

    def test_perplexity_known_values(self):
        # e^1 ~= 2.7183, e^2 ~= 7.3891
        assert math.isclose(compute_perplexity(1.0), round(math.e, 4), rel_tol=1e-4)
        assert math.isclose(compute_perplexity(2.0), round(math.exp(2.0), 4), rel_tol=1e-4)

    def test_perplexity_large_loss_clamped(self):
        # Loss of 100 should be clamped to 50 and not overflow
        ppl = compute_perplexity(100.0)
        assert ppl > 1e20
        assert not math.isinf(ppl)

    def test_perplexity_negative_loss_handled(self):
        # Even if loss is slightly negative due to precision, exp is finite
        ppl = compute_perplexity(-0.5)
        assert 0.0 < ppl < 1.0


class TestDistinctMetrics:
    def test_distinct_1_all_identical(self):
        tokens = ["apple", "apple", "apple", "apple"]
        assert compute_distinct_n(tokens, n=1) == 0.25

    def test_distinct_1_all_unique(self):
        tokens = ["apple", "banana", "cherry", "date"]
        assert compute_distinct_n(tokens, n=1) == 1.0

    def test_distinct_n_empty_or_short(self):
        assert compute_distinct_n([], n=1) == 0.0
        assert compute_distinct_n(["one"], n=2) == 0.0

    def test_distinct_2_repeated_pairs(self):
        tokens = ["a", "b", "a", "b", "a", "b"]
        # Bigrams: (a, b), (b, a), (a, b), (b, a), (a, b) -> total 5, unique: (a, b), (b, a)
        distinct_2 = compute_distinct_n(tokens, n=2)
        assert math.isclose(distinct_2, 2 / 5, abs_tol=1e-3)

    def test_evaluate_text_diversity_dictionary_keys(self):
        text = "To be, or not to be, that is the question."
        diversity = evaluate_text_diversity(text)
        assert "distinct_1_word" in diversity
        assert "distinct_2_word" in diversity
        assert "distinct_1_char" in diversity
        assert "distinct_2_char" in diversity
        assert 0.0 <= diversity["distinct_1_word"] <= 1.0
        assert 0.0 <= diversity["distinct_2_word"] <= 1.0


class TestRepetitionMetrics:
    def test_repetition_rate_zero_when_unique(self):
        tokens = ["alpha", "beta", "gamma", "delta"]
        rate = repetition_rate(tokens, n=2)
        assert rate == 0.0

    def test_repetition_rate_high_when_completely_repetitive(self):
        tokens = ["ha"] * 10
        # 1-gram: 1 unique out of 10 -> repetition rate = (10 - 1) / 10 = 0.9
        rate = repetition_rate(tokens, n=1)
        assert math.isclose(rate, 0.9, rel_tol=1e-4)

    def test_repetition_rate_short_sequence(self):
        tokens = ["single"]
        assert repetition_rate(tokens, n=3) == 0.0

    def test_analyze_repetition_multi_order(self):
        repeated_text = "the king and the queen and the prince and the king"
        rep_dict = analyze_repetition(repeated_text, n_values=(2, 3))
        assert "word_rep_rate_2gram" in rep_dict
        assert "word_rep_rate_3gram" in rep_dict
        assert "char_rep_rate_2gram" in rep_dict
        assert "char_rep_rate_3gram" in rep_dict
        assert rep_dict["word_rep_rate_2gram"] > 0.0
