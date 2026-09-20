"""
Text cleaner module for Day 104 Semantic Search Engine.
Handles normalization, punctuation removal, and stopword management.
"""

import re
import string
from typing import Optional, Set


class TextCleaner:
    """Text normalization and cleaning utility."""

    # Default common English stop words
    DEFAULT_STOPWORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    }

    def __init__(self, stopwords: Optional[Set[str]] = None, remove_stopwords: bool = False):
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.remove_stopwords = remove_stopwords

    def clean(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        strip_whitespace: bool = True
    ) -> str:
        """Clean and normalize raw text.
        
        Args:
            text: Input string.
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to replace punctuation with spaces.
            strip_whitespace: Whether to collapse multiple whitespace characters.
            
        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        if lowercase:
            text = text.lower()

        if remove_punctuation:
            # Replace punctuation with space to prevent concatenating adjacent words
            text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        if strip_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        if self.remove_stopwords and self.stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = " ".join(words)

        return text
