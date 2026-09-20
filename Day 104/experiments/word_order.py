"""
Experiment 4: The Word Order Problem in Bag-of-Vectors Mean Pooling.
"""

import sys
from pathlib import Path
import numpy as np

# Add app parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import config
from app.data.loader import DataLoader
from app.preprocessing.cleaner import TextCleaner
from app.preprocessing.tokenizer import Tokenizer
from app.representations.document_embedding import DocumentEmbedder
from app.retrieval.ranking import cosine_similarity


def run_experiment():
    print("=" * 65)
    print("EXPERIMENT 4: WORD ORDER LIMITATION IN MEAN POOLING")
    print("=" * 65)

    docs = DataLoader.load_documents(config.DOCUMENTS_PATH)
    cleaner = TextCleaner()
    tokenizer = Tokenizer()

    tokenized_corpus = [
        tokenizer.tokenize(cleaner.clean(f"{d['title']} {d['text']}"))
        for d in docs
    ]

    embedder = DocumentEmbedder(embedding_dim=64, seed=42)
    embedder.train_on_corpus(tokenized_corpus)

    sentence_a = "the dog chased the cat"
    sentence_b = "the cat chased the dog"

    tokens_a = tokenizer.tokenize(cleaner.clean(sentence_a))
    tokens_b = tokenizer.tokenize(cleaner.clean(sentence_b))

    vec_a = embedder.embed_document(tokens_a, pooling="mean")
    vec_b = embedder.embed_document(tokens_b, pooling="mean")

    sim = cosine_similarity(vec_a, vec_b)

    print(f"\nSentence A: '{sentence_a}'")
    print(f"Sentence B: '{sentence_b}'")
    print(f"Tokens A:   {tokens_a}")
    print(f"Tokens B:   {tokens_b}")
    print(f"Cosine Similarity (Mean Pooling): {sim:.4f}")

    print("\nAdversarial Retrieval Test on Corpus:")
    doc_36 = next(d for d in docs if d["id"] == 36)
    doc_43 = next(d for d in docs if d["id"] == 43)

    print(f"Doc 36 (Dog chases cat): '{doc_36['title']}'")
    print(f"Doc 43 (Cat chases dog): '{doc_43['title']}'")

    vec_36 = embedder.embed_document(tokenizer.tokenize(cleaner.clean(f"{doc_36['title']} {doc_36['text']}")))
    vec_43 = embedder.embed_document(tokenizer.tokenize(cleaner.clean(f"{doc_43['title']} {doc_43['text']}")))

    print(f"\nQuery: '{sentence_a}' (Dog chased cat)")
    print(f"  Similarity with Doc 36: {cosine_similarity(vec_a, vec_36):.4f}")
    print(f"  Similarity with Doc 43: {cosine_similarity(vec_a, vec_43):.4f}")

    print(f"\nQuery: '{sentence_b}' (Cat chased dog)")
    print(f"  Similarity with Doc 36: {cosine_similarity(vec_b, vec_36):.4f}")
    print(f"  Similarity with Doc 43: {cosine_similarity(vec_b, vec_43):.4f}")

    print("\nConclusion:")
    print("  -> Mean word embeddings completely lose sequence and syntactic order.")
    print("  -> The similarity between opposite agent-patient sentences is virtually identical.")
    print("  -> This is a fundamental limitation of static bag-of-vectors pooling, which motivates modern Transformers (BERT/GPT) with positional encodings.")


if __name__ == "__main__":
    run_experiment()
