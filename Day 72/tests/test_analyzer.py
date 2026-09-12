"""
Unit tests for relationship analyzer and business Q&A.
"""
import pytest
import pandas as pd
import numpy as np
from app.analyzer import RelationshipAnalyzer
from app.correlation_matrix import compute_all_pairwise_results

def test_analyzer_top_positive_and_negative(sample_numeric_df):
    pairwise_df = compute_all_pairwise_results(sample_numeric_df)
    analyzer = RelationshipAnalyzer(pairwise_df, sample_numeric_df)
    
    top_pos = analyzer.get_top_positive(2)
    assert len(top_pos) <= 2
    assert (top_pos["Pearson_r"] > 0).all()
    
    top_neg = analyzer.get_top_negative(2)
    assert len(top_neg) <= 2
    assert (top_neg["Pearson_r"] < 0).all()

def test_analyzer_multicollinearity_detection(sample_numeric_df):
    pairwise_df = compute_all_pairwise_results(sample_numeric_df)
    analyzer = RelationshipAnalyzer(pairwise_df, sample_numeric_df)
    mc = analyzer.get_multicollinear_pairs()
    assert isinstance(mc, pd.DataFrame)

def test_analyzer_divergent_pairs(sample_numeric_df):
    pairwise_df = compute_all_pairwise_results(sample_numeric_df)
    analyzer = RelationshipAnalyzer(pairwise_df, sample_numeric_df)
    div = analyzer.get_divergent_pairs()
    assert isinstance(div, pd.DataFrame)

def test_analyzer_answers_10_questions(sample_numeric_df):
    pairwise_df = compute_all_pairwise_results(sample_numeric_df)
    analyzer = RelationshipAnalyzer(pairwise_df, sample_numeric_df)
    q_and_a = analyzer.answer_business_questions()
    assert len(q_and_a) == 10
    for item in q_and_a:
        assert "Question" in item and "Answer" in item
        assert len(item["Answer"]) > 0
