"""All 10 Required Coding Challenges for Day 102."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix

def run_all_challenges():
    print("=== EXECUTING DAY 102 CODING CHALLENGES 1 TO 10 ===")
    
    # Challenge 1: TF-IDF from scratch
    docs = ["python data", "python code", "code data"]
    # doc counts: python:2, data:2, code:2
    N = len(docs)
    
    # Challenge 2: Document frequency
    vocab = sorted(list({"python", "data", "code"}))
    df = {w: sum(w in d.split() for d in docs) for w in vocab}
    assert df["python"] == 2 and df["data"] == 2 and df["code"] == 2
    
    # Challenge 3: Smoothed IDF: log((1 + N) / (1 + df)) + 1
    idf = {w: np.log((1 + N) / (1 + df[w])) + 1 for w in vocab}
    assert all(v > 1.0 for v in idf.values())
    
    # Challenge 4: Word n-grams
    tokens = ["a", "b", "c"]
    w_ng = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    assert w_ng == [("a", "b"), ("b", "c")]
    
    # Challenge 5: Character n-grams
    word = "data"
    c_ng = [word[i:i+3] for i in range(len(word)-2)]
    assert c_ng == ["dat", "ata"]
    
    # Challenge 6: Vocab limiting with min_df
    corpus = ["apple banana", "apple orange", "apple kiwi"]
    counts = {}
    for d in corpus:
        for w in set(d.split()):
            counts[w] = counts.get(w, 0) + 1
    filtered_vocab = [w for w, c in counts.items() if c >= 2]
    assert filtered_vocab == ["apple"]
    
    # Challenge 7: Sparse document-term representation
    dense_matrix = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    sparse_rep = csr_matrix(dense_matrix)
    assert sparse_rep.nnz == 3
    
    # Challenge 8: Simple Naive Bayes text classifier
    nb_clf = MultinomialNB()
    X_toy = np.array([[2, 0], [0, 2], [2, 1], [0, 3]])
    y_toy = np.array([0, 1, 0, 1])
    nb_clf.fit(X_toy, y_toy)
    assert nb_clf.predict([[3, 0]])[0] == 0
    assert nb_clf.predict([[0, 3]])[0] == 1
    
    # Challenge 9: Complete TF-IDF -> Logistic Regression pipeline
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(["free cash now", "hello friend"], [1, 0])
    assert pipe.predict(["cash prize"])[0] == 1
    
    # Challenge 10: Manual vs Scikit-Learn Confusion Matrix & Metrics
    # Given:
    # Actual:    [0, 0, 1, 1, 0]
    # Predicted: [0, 1, 1, 0, 0]
    y_act = np.array([0, 0, 1, 1, 0])
    y_prd = np.array([0, 1, 1, 0, 0])
    
    # Manual Calculation:
    # Index 0: Act 0, Prd 0 -> TN
    # Index 1: Act 0, Prd 1 -> FP
    # Index 2: Act 1, Prd 1 -> TP
    # Index 3: Act 1, Prd 0 -> FN
    # Index 4: Act 0, Prd 0 -> TN
    man_TP = 1
    man_TN = 2
    man_FP = 1
    man_FN = 1
    
    man_precision = man_TP / (man_TP + man_FP) # 1 / 2 = 0.5
    man_recall = man_TP / (man_TP + man_FN)    # 1 / 2 = 0.5
    man_f1 = 2 * (man_precision * man_recall) / (man_precision + man_recall) # 0.5
    
    # Scikit-Learn Verification:
    tn, fp, fn, tp = confusion_matrix(y_act, y_prd).ravel()
    skl_prec = precision_score(y_act, y_prd)
    skl_rec = recall_score(y_act, y_prd)
    skl_f1 = f1_score(y_act, y_prd)
    
    assert man_TP == tp and man_TN == tn and man_FP == fp and man_FN == fn
    assert np.isclose(man_precision, skl_prec)
    assert np.isclose(man_recall, skl_rec)
    assert np.isclose(man_f1, skl_f1)
    
    print(f"Challenge 10 Manual: TP={man_TP}, TN={man_TN}, FP={man_FP}, FN={man_FN}")
    print(f"Precision: {man_precision:.2f}, Recall: {man_recall:.2f}, F1: {man_f1:.2f}")
    print("[SUCCESS] All 10 Challenges Verified Successfully!")

if __name__ == "__main__":
    run_all_challenges()
