"""Task 1 & Challenge 3: Text Cleaner implementation."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import re
import string

def normalize_text(text: str) -> str:
    """
    Normalizes input string:
    - handles lowercase
    - removes punctuation
    - trims whitespace
    - safely handles empty strings & non-string types
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
        
    text = text.lower().strip()
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    return re.sub(r'\s+', ' ', text).strip()

def run_challenge():
    sample = "  HELLO World!!!  "
    cleaned = normalize_text(sample)
    print(f"Original: '{sample}' -> Cleaned: '{cleaned}'")
    assert cleaned == "hello world"
    print("[SUCCESS] Text Cleaner challenge passed.")

if __name__ == "__main__":
    run_challenge()
