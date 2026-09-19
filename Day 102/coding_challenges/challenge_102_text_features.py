"""Challenge: Text feature extraction with min_df and max_features."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from sklearn.feature_extraction.text import TfidfVectorizer

def run():
    docs = [
        "free cash prize now",
        "claim your cash bonus",
        "free tickets to movie",
        "call now to claim prize"
    ]
    vec = TfidfVectorizer(min_df=2, max_features=5)
    X = vec.fit_transform(docs)
    vocab = vec.vocabulary_
    print("Vocabulary with min_df=2, max_features=5:", vocab)
    assert len(vocab) <= 5
    for word in vocab:
        df_count = sum(word in d for d in docs)
        assert df_count >= 2
    print("[SUCCESS] Text feature extraction parameters verified.")

if __name__ == "__main__":
    run()
