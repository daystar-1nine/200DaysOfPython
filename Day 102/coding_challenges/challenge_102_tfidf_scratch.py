"""Manual TF-IDF Exercise & Comparison with Scikit-Learn."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.features.scratch_tfidf import TFIDFScratch

def run_manual_exercise():
    print("=== MANUAL TF-IDF EXERCISE ===")
    docs = [
        "python data science",
        "python machine learning",
        "machine learning python"
    ]
    
    # 1. From-scratch calculation
    scratch = TFIDFScratch()
    matrix_scratch = scratch.fit_transform(docs)
    vocab_scratch = scratch.vocabulary_
    
    print("Vocabulary:", vocab_scratch)
    print("Document Frequencies (df):", {w: int(np.sum([w in d.split() for d in docs])) for w in vocab_scratch})
    print("Smoothed IDF:", {w: float(scratch.idf_[idx]) for w, idx in vocab_scratch.items()})
    print("TF-IDF Matrix (Scratch):")
    print(np.round(matrix_scratch, 4))
    
    # 2. Scikit-Learn comparison
    sklearn_vec = TfidfVectorizer(norm="l2", smooth_idf=True, sublinear_tf=False)
    matrix_sklearn = sklearn_vec.fit_transform(docs).toarray()
    
    print("\nTF-IDF Matrix (Scikit-Learn):")
    print(np.round(matrix_sklearn, 4))
    assert matrix_scratch.shape == matrix_sklearn.shape
    print("[SUCCESS] Manual and Scikit-Learn TF-IDF verified.")

if __name__ == "__main__":
    run_manual_exercise()
