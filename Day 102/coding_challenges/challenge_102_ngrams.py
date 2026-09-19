"""Challenge: Word and Character N-grams."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def word_ngrams(tokens, n=2):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []

def char_ngrams(text, n=3):
    text = text.strip()
    return [text[i : i + n] for i in range(len(text) - n + 1)] if len(text) >= n else []

def run():
    tokens = ["machine", "learning", "engineer"]
    w_bi = word_ngrams(tokens, 2)
    c_tri = char_ngrams("learning", 3)
    
    print("Tokens:", tokens)
    print("Word Bigrams:", w_bi)
    print("Character Trigrams ('learning'):", c_tri)
    
    assert w_bi == [("machine", "learning"), ("learning", "engineer")]
    assert "lea" in c_tri and "ing" in c_tri
    print("[SUCCESS] Word & Character N-grams verified.")

if __name__ == "__main__":
    run()
