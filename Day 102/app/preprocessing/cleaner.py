"""Different cleaning strategies for empirical comparison."""
from app.preprocessing.normalization import normalize_text

def clean_minimal(text: str) -> str:
    """Strategy A: Minimal preprocessing (lowercase & whitespace normalization)."""
    return normalize_text(text, lowercase=True, remove_punct=False, remove_urls=False, remove_numbers=False)

def clean_punctuation(text: str) -> str:
    """Strategy B: Punctuation removal & URL stripping."""
    return normalize_text(text, lowercase=True, remove_punct=True, remove_urls=True, remove_numbers=False)

def clean_aggressive(text: str) -> str:
    """Strategy C: Aggressive preprocessing (removes punctuation, URLs, numbers, short words)."""
    normalized = normalize_text(text, lowercase=True, remove_punct=True, remove_urls=True, remove_numbers=True)
    tokens = [w for w in normalized.split() if len(w) > 2]
    return " ".join(tokens)
