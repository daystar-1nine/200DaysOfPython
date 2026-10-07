"""
Unit tests for BLEU and ROUGE n-gram and longest common subsequence metrics.
"""
import pytest
from app.evaluators.bleu import (
    compute_ngrams,
    compute_brevity_penalty,
    modified_ngram_precision,
    bleu_score,
    sentence_bleu,
    corpus_bleu
)
from app.evaluators.rouge import (
    compute_lcs_length,
    rouge_n,
    rouge_l,
    compute_rouge_all
)


def test_compute_ngrams_1gram():
    tokens = ["the", "quick", "brown", "fox"]
    ngrams = compute_ngrams(tokens, 1)
    assert ngrams[("the",)] == 1
    assert ngrams[("fox",)] == 1
    assert len(ngrams) == 4


def test_compute_ngrams_2gram():
    tokens = ["the", "cat", "in", "the", "hat"]
    ngrams = compute_ngrams(tokens, 2)
    assert ngrams[("the", "cat")] == 1
    assert ngrams[("cat", "in")] == 1
    assert ngrams[("in", "the")] == 1
    assert ngrams[("the", "hat")] == 1
    assert len(ngrams) == 4


def test_compute_ngrams_empty_or_too_short():
    assert len(compute_ngrams([], 1)) == 0
    assert len(compute_ngrams(["word"], 2)) == 0


def test_compute_brevity_penalty_longer_candidate():
    # c > r -> BP == 1.0
    assert compute_brevity_penalty(10, 8) == 1.0
    assert compute_brevity_penalty(10, 10) == 1.0


def test_compute_brevity_penalty_shorter_candidate():
    # c < r -> BP < 1.0
    bp = compute_brevity_penalty(5, 10)
    assert bp < 1.0
    assert bp == pytest.approx(0.3678, abs=1e-3)


def test_compute_brevity_penalty_zero_candidate():
    assert compute_brevity_penalty(0, 10) == 0.0


def test_modified_ngram_precision_clipping():
    # candidate repeats "the" 4 times, but reference only has it twice
    cand = ["the", "the", "the", "the"]
    refs = [["the", "the", "cat"]]
    matches, total = modified_ngram_precision(cand, refs, 1)
    assert matches == 2  # clipped at 2
    assert total == 4


def test_sentence_bleu_identical_text():
    text = "Machine learning models optimize parameters using gradient descent."
    res = sentence_bleu(text, text)
    assert res["bleu_1"] == 1.0
    assert res["bleu_4"] == 1.0
    assert res["bleu_overall"] == 1.0


def test_sentence_bleu_disjoint_text():
    pred = "quantum entanglement in superconductivity"
    ref = "database indexing using b trees"
    # Strict without smoothing is 0.0
    strict_score = bleu_score(pred, ref, smoothing=False)
    assert strict_score == 0.0
    res = sentence_bleu(pred, ref)
    assert res["bleu_overall"] < 0.20


def test_sentence_bleu_multi_reference():
    pred = "A dog was running in the park."
    refs = [
        "A canine played on the lawn.",
        "The dog ran across the open park."
    ]
    res = sentence_bleu(pred, refs)
    assert res["bleu_1"] > 0.3
    assert res["bleu_overall"] >= 0.0


def test_corpus_bleu_aggregate():
    preds = [
        "Paris is the capital of France.",
        "Python is a programming language."
    ]
    refs = [
        "Paris is the capital of France.",
        "Python is a programming language."
    ]
    res = corpus_bleu(preds, refs)
    assert res["bleu_overall"] == 1.0


def test_compute_lcs_length_identical():
    s1 = ["a", "b", "c", "d"]
    s2 = ["a", "b", "c", "d"]
    assert compute_lcs_length(s1, s2) == 4


def test_compute_lcs_length_partial_subsequence():
    s1 = ["the", "quick", "brown", "fox", "jumps"]
    s2 = ["the", "brown", "fox", "leaps"]
    # Common subsequence: ["the", "brown", "fox"] -> length 3
    assert compute_lcs_length(s1, s2) == 3


def test_compute_lcs_length_disjoint():
    s1 = ["red", "green", "blue"]
    s2 = ["yellow", "black", "orange"]
    assert compute_lcs_length(s1, s2) == 0


def test_compute_lcs_length_empty():
    assert compute_lcs_length([], ["a", "b"]) == 0
    assert compute_lcs_length(["a"], []) == 0


def test_rouge_1_identical_text():
    text = "Relational databases use SQL for queries"
    res = rouge_n(text, text, n=1)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_rouge_1_disjoint_text():
    res = rouge_n("apple orange banana", "cat dog elephant", n=1)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_rouge_2_bigram_overlap():
    pred = "the cat sat on the mat"
    ref = "the cat lay on the mat"
    res = rouge_n(pred, ref, n=2)
    assert res["precision"] > 0.25
    assert res["recall"] > 0.25
    assert res["f1"] > 0.25


def test_rouge_l_subsequence_f1():
    pred = "Python is a dynamic interpreted language"
    ref = "Python is an interpreted programming language"
    res = rouge_l(pred, ref)
    assert res["f1"] > 0.6
    assert res["recall"] > 0.6


def test_compute_rouge_all_multi_reference():
    pred = "Docker containers isolate processes"
    refs = [
        "Containers isolate application workloads",
        "Docker packages dependencies into portable containers"
    ]
    res = compute_rouge_all(pred, refs)
    assert "rouge_1" in res
    assert "rouge_2" in res
    assert "rouge_l" in res
    assert res["rouge_1"] > 0.0
