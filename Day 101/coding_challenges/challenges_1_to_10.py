"""Implementation and verification of all 10 coding challenges."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import string
import re
from collections import Counter
from typing import List, Tuple, Dict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Challenge 1: Reverse sentence word-by-word
def challenge_1_reverse_words(sentence: str) -> str:
    return " ".join(sentence.strip().split()[::-1])

# Challenge 2: Count word frequencies
def challenge_2_word_frequencies(tokens: List[str]) -> Counter:
    return Counter(tokens)

# Challenge 3: Remove punctuation
def challenge_3_remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

# Challenge 4: Implement tokenizer
def challenge_4_tokenize(text: str) -> List[str]:
    cleaned = challenge_3_remove_punctuation(text.lower().strip())
    return cleaned.split()

# Challenge 5: Implement n-grams
def challenge_5_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []

# Challenge 6: Build vocabulary
def challenge_6_build_vocab(tokenized_docs: List[List[str]]) -> Dict[str, int]:
    all_words = sorted(list({w for doc in tokenized_docs for w in doc}))
    return {w: i for i, w in enumerate(all_words)}

# Challenge 7: Bag of Words document vector
def challenge_7_bow_vector(doc_tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    vec = [0] * len(vocab)
    for t in doc_tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    return vec

# Challenge 8: Find top-N words
def challenge_8_top_n_words(tokens: List[str], n: int = 5) -> List[Tuple[str, int]]:
    return Counter(tokens).most_common(n)

# Challenge 9: Calculate document lengths
def challenge_9_doc_lengths(documents: List[str]) -> List[Dict[str, int]]:
    return [{"chars": len(doc), "words": len(doc.split())} for doc in documents]

# Challenge 10: Complete text classification pipeline
def challenge_10_classification_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(random_state=42))
    ])

def test_all_challenges():
    print("Testing All 10 Coding Challenges...")
    
    # C1
    assert challenge_1_reverse_words("I love Python") == "Python love I"
    
    # C2
    c2 = challenge_2_word_frequencies(["python", "data", "python"])
    assert c2["python"] == 2 and c2["data"] == 1
    
    # C3
    assert challenge_3_remove_punctuation("Hello, World!") == "Hello World"
    
    # C4
    assert challenge_4_tokenize("Hello, World!") == ["hello", "world"]
    
    # C5
    assert challenge_5_ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    
    # C6
    vocab = challenge_6_build_vocab([["cat", "dog"], ["cat", "bird"]])
    assert vocab == {"bird": 0, "cat": 1, "dog": 2}
    
    # C7
    bow_vec = challenge_7_bow_vector(["cat", "cat", "bird"], vocab)
    assert bow_vec == [1, 2, 0]
    
    # C8
    top = challenge_8_top_n_words(["a", "b", "a", "c", "a", "b"], 2)
    assert top == [("a", 3), ("b", 2)]
    
    # C9
    lens = challenge_9_doc_lengths(["Hello world", "Test"])
    assert lens[0]["words"] == 2 and lens[1]["words"] == 1
    
    # C10
    pipe = challenge_10_classification_pipeline()
    pipe.fit(["free cash now", "hey are we meeting"], [1, 0])
    pred = pipe.predict(["win cash prize"])
    assert pred[0] == 1
    
    print("[SUCCESS] All 10 Coding Challenges successfully verified!")

if __name__ == "__main__":
    test_all_challenges()
