"""
RAG generation quality evaluators: Answer Faithfulness, Context Relevance, and Hallucination auditing.
"""
from typing import Dict, Any, List, Set
from app.ingestion.cleaner import tokenize_words
from app.generation.generator import UNSUPPORTED_RESPONSE


def compute_faithfulness(answer: str, context: str) -> float:
    """
    Computes answer faithfulness:
    What fraction of informational tokens in the answer are grounded in the retrieved context?
    Returns float in range [0.0, 1.0].
    """
    if not answer or not answer.strip():
        return 0.0

    # If the answer is the canonical unsupported response, faithfulness is 1.0 (truthful abstention)
    if UNSUPPORTED_RESPONSE.lower() in answer.lower():
        return 1.0

    ans_tokens = tokenize_words(answer)
    ctx_tokens = set(tokenize_words(context))

    # Strip conversational stopwords
    stops = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "to", "for", "of", "and", "or", "it", "this", "that", "sources", "source", "chunk"}
    content_tokens = [t for t in ans_tokens if t not in stops]

    if not content_tokens:
        return 1.0

    grounded_count = sum(1 for t in content_tokens if t in ctx_tokens)
    return float(grounded_count / len(content_tokens))


def compute_context_relevance(question: str, context: str) -> float:
    """
    Computes context relevance:
    What fraction of key terms in the user question are addressed by the retrieved context?
    """
    if not question or not context:
        return 0.0

    q_tokens = tokenize_words(question)
    ctx_tokens = set(tokenize_words(context))

    stops = {"what", "is", "are", "how", "why", "the", "a", "an", "in", "of", "to", "explain", "describe", "does", "do"}
    key_q = [t for t in q_tokens if t not in stops]

    if not key_q:
        return 1.0

    hits = sum(1 for t in key_q if t in ctx_tokens)
    return float(hits / len(key_q))


def compute_token_overlap_f1(pred: str, target: str) -> float:
    """Computes token-level precision, recall, and harmonic mean F1 between two text sequences."""
    p_tokens = tokenize_words(pred)
    t_tokens = tokenize_words(target)
    if not p_tokens or not t_tokens:
        return 0.0

    common = set(p_tokens) & set(t_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(p_tokens)
    recall = len(common) / len(t_tokens)
    return float(2 * precision * recall / (precision + recall))


def evaluate_grounding(
    question: str,
    answer: str,
    context: str,
    is_supported: bool
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of an individual RAG generated response:
    - faithfulness: degree of support by context
    - context_relevance: degree to which context addresses question
    - is_hallucinated: flag indicating presence of fabricated or ungrounded facts
    """
    faith = compute_faithfulness(answer, context)
    relevance = compute_context_relevance(question, context)

    # Hallucination check
    is_unsupported_reply = UNSUPPORTED_RESPONSE.lower() in answer.lower()
    if is_supported:
        # Expected supported: if answer is truthful and faithful >= 0.70, not hallucinated
        is_hallucinated = bool(faith < 0.60 and not is_unsupported_reply)
    else:
        # Expected unsupported: answering with confident ungrounded claims is a hallucination
        is_hallucinated = not is_unsupported_reply

    return {
        "faithfulness": round(faith, 4),
        "context_relevance": round(relevance, 4),
        "is_hallucinated": is_hallucinated,
        "is_grounded": bool(faith >= 0.70 or is_unsupported_reply)
    }
