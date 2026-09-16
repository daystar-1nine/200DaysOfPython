
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def get_churn_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])

def get_text_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", MultinomialNB(alpha=1.0))
    ])
