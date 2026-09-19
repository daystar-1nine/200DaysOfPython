"""Extracts vocabulary and coefficient insights from trained models."""
from typing import List, Tuple
import numpy as np
from sklearn.pipeline import Pipeline

def get_top_features_by_class(pipeline: Pipeline, top_n: int = 15) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Extracts top positive (spam) and negative (ham) features from Logistic Regression."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    
    # Top Spam (highest positive coefficients)
    top_spam_indices = np.argsort(coefs)[-top_n:][::-1]
    top_spam = [(feature_names[i], float(coefs[i])) for i in top_spam_indices]
    
    # Top Ham (lowest negative coefficients)
    top_ham_indices = np.argsort(coefs)[:top_n]
    top_ham = [(feature_names[i], float(coefs[i])) for i in top_ham_indices]
    
    return top_spam, top_ham
