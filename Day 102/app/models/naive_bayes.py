"""Multinomial Naive Bayes model factory."""
from sklearn.naive_bayes import MultinomialNB

def get_naive_bayes_model(alpha: float = 1.0) -> MultinomialNB:
    """Returns configured MultinomialNB."""
    return MultinomialNB(alpha=alpha)
