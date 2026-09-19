"""Multinomial Naive Bayes text classification model."""
from sklearn.naive_bayes import MultinomialNB

def get_naive_bayes_model(alpha: float = 1.0) -> MultinomialNB:
    """Returns configured MultinomialNB model."""
    return MultinomialNB(alpha=alpha)
