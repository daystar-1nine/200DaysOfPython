"""Task 5 & Challenge 7: Bag of Words from scratch vs scikit-learn."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.features.bow import BagOfWords

def run_challenge():
    docs = [
        "I love Python",
        "I love Data Science"
    ]
    
    # Scratch implementation
    bow = BagOfWords()
    dtm_scratch = bow.fit_transform(docs)
    vocab_scratch = bow.vocabulary_
    
    print("--- From Scratch Bag of Words ---")
    print(f"Vocabulary: {vocab_scratch}")
    print("Document-Term Matrix:")
    print(dtm_scratch)
    
    # Scikit-learn comparison
    sklearn_cv = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')
    dtm_sklearn = sklearn_cv.fit_transform(docs).toarray()
    vocab_sklearn = sklearn_cv.vocabulary_
    
    print("\n--- Scikit-Learn CountVectorizer ---")
    print(f"Vocabulary: {vocab_sklearn}")
    print("Document-Term Matrix:")
    print(dtm_sklearn)
    
    # Verification
    assert dtm_scratch.shape == dtm_sklearn.shape, "Shape mismatch between scratch and sklearn"
    print("\n[SUCCESS] Scratch Bag of Words verification successful!")

if __name__ == "__main__":
    run_challenge()
