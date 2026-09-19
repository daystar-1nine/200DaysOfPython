"""Extracts top predictive feature coefficients."""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict

def extract_top_features(pipeline: Pipeline, top_n: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts top positive (spam) and negative (ham) features from linear models.
    Supports LogisticRegression and LinearSVC.
    """
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    
    # Handle calibrated classifier
    if hasattr(clf, "calibrated_classifiers_"):
        base = clf.calibrated_classifiers_[0].estimator
        coef = base.coef_[0]
    elif hasattr(clf, "coef_"):
        coef = clf.coef_[0]
    else:
        # For Naive Bayes, use feature_log_prob_ difference
        coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        
    feature_names = np.array(vec.get_feature_names_out())
    
    # Top Spam
    top_spam_idx = np.argsort(coef)[-top_n:][::-1]
    df_spam = pd.DataFrame({
        "feature": feature_names[top_spam_idx],
        "coefficient": coef[top_spam_idx],
        "association": "spam"
    })
    
    # Top Ham
    top_ham_idx = np.argsort(coef)[:top_n]
    df_ham = pd.DataFrame({
        "feature": feature_names[top_ham_idx],
        "coefficient": coef[top_ham_idx],
        "association": "ham"
    })
    
    return df_spam, df_ham
