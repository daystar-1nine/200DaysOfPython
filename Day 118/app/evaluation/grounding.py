"""
Grounding and Faithfulness Evaluation for RAG Generation.
Evaluates context citation precision, claim-context entailment, and hallucination rate.
"""
from typing import List, Dict, Set
import re

from app.ingestion.cleaner import tokenize_words


def compute_citation_precision(citations: List[Any], context_text: str) -> float:
    """Computes fraction of citations whose snippets are authentically present in context."""
    if not citations:
        return 1.0

    context_lower = context_text.lower()
    valid = 0
    for c in citations:
        snippet_raw = getattr(c, "snippet", str(c)).replace("...", "").strip().lower()
        if snippet_raw and snippet_raw[:40] in context_lower:
            valid += 1
        elif not snippet_raw:
            valid += 1

    return valid / len(citations)


def evaluate_faithfulness(answer: str, context_text: str) -> float:
    """
    Evaluates what proportion of significant content words in the generated answer
    are directly supported by the retrieved context.
    """
    if not answer or not context_text:
        return 0.0

    ans_tokens = set(tokenize_words(answer))
    ctx_tokens = set(tokenize_words(context_text))

    stopwords = {
        "is", "the", "a", "an", "and", "or", "to", "in", "of", "based", "on", "provided",
        "documentation", "source", "chunk", "that", "this", "it", "for", "with", "as"
    }
    content_tokens = ans_tokens - stopwords
    if not content_tokens:
        return 1.0

    supported = sum(1 for t in content_tokens if t in ctx_tokens)
    return supported / len(content_tokens)
