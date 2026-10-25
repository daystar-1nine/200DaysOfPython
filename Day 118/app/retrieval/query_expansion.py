"""
Query Expansion, Query Rewriting, and Multi-Query Retrieval Engine.
Improves retrieval recall by bridging the vocabulary mismatch problem between user phrasing and indexed documents.
"""
from typing import List, Dict, Set, Optional, Callable
import re

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk
from app.retrieval.rrf import reciprocal_rank_fusion


# Technical domain synonym dictionary
TECHNICAL_SYNONYMS: Dict[str, List[str]] = {
    "overfitting": ["generalization", "validation loss", "regularization", "dropout", "early stopping"],
    "underfitting": ["high bias", "model capacity", "training loss", "feature engineering"],
    "memory leak": ["garbage collection", "ram consumption", "uncollected objects", "reference cycle"],
    "http 404": ["not found", "status code", "client error", "url routing", "endpoint"],
    "inner join": ["sql join", "relational algebra", "matching rows", "foreign key", "table merge"],
    "gradient descent": ["learning rate", "optimization", "backpropagation", "loss function", "sgd"],
    "tf-idf": ["term frequency", "inverse document frequency", "lexical weight", "text vectorization"],
    "bm25": ["okapi bm25", "probabilistic relevance", "term frequency saturation", "document length normalization"],
    "list comprehension": ["generator expression", "python syntax", "iterable loop", "syntactic sugar"],
    "sqlstate 23505": ["unique constraint violation", "duplicate key", "primary key collision", "integrity error"],
    "deadlock": ["mutual exclusion", "lock contention", "circular wait", "concurrency", "transaction"],
    "indexing": ["b-tree", "hash index", "lookup performance", "query optimization", "composite index"],
    "cross entropy": ["loss function", "log loss", "classification", "softmax probability"],
    "rag": ["retrieval augmented generation", "vector search", "knowledge grounding", "external context"],
    "embedding": ["vector representation", "dense semantic metric", "latent space", "cosine similarity"],
    "attention": ["self attention", "transformer", "multihead", "query key value", "cross attention"],
}

# Conversational boilerplate patterns to strip during query rewriting
CONVERSATIONAL_PREFIXES = [
    r"^can you (please )?(tell|explain|show|give) (me )?(what is|about|how to)?\s*",
    r"^could you (please )?(clarify|describe|detail)?\s*",
    r"^i (want|would like) to know (about|how to)?\s*",
    r"^what is the (difference between|meaning of|definition of)\s*",
    r"^how (can|do) (i|we|one)\s*",
    r"^please (explain|describe|show)\s*",
    r"^tell me (about|how)\s*",
    r"^what (is|are)\s*",
]


def expand_query(query: str, max_synonyms: int = 3) -> str:
    """
    Expands a query by detecting technical terms and appending domain synonyms.
    Example:
        'overfitting in neural networks' -> 'overfitting in neural networks generalization validation loss regularization'
    """
    if not query or not query.strip():
        return ""

    query_lower = query.lower()
    expanded_terms: List[str] = []
    seen: Set[str] = set(re.findall(r"\b\w+\b", query_lower))

    for term, syns in TECHNICAL_SYNONYMS.items():
        if term in query_lower:
            for s in syns[:max_synonyms]:
                s_words = s.split()
                if not any(w in seen for w in s_words):
                    expanded_terms.append(s)
                    seen.update(s_words)

    if not expanded_terms:
        return query.strip()

    return f"{query.strip()} {' '.join(expanded_terms)}"


def rewrite_query(query: str) -> str:
    """
    Rewrites a conversational or verbose query into a concise, keyword-dense search query.
    Example:
        'Can you please tell me how to prevent a model from memorizing training data?'
        -> 'prevent model memorizing training data'
    """
    if not query or not query.strip():
        return ""

    cleaned = query.strip()
    for pattern in CONVERSATIONAL_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing punctuation
    cleaned = re.sub(r"[?!.,;]+$", "", cleaned).strip()

    # Strip filler words if string is long enough
    tokens = cleaned.split()
    if len(tokens) > 3:
        fillers = {"a", "an", "the", "in", "of", "to", "for", "with", "on", "at", "by", "from"}
        meaningful = [t for t in tokens if t.lower() not in fillers or t.lower() in TECHNICAL_SYNONYMS]
        if meaningful:
            cleaned = " ".join(meaningful)

    return cleaned if cleaned else query.strip()


def generate_subqueries(query: str, count: int = 3) -> List[str]:
    """
    Generates multifaceted subqueries representing diverse retrieval perspectives:
    1. Direct / original rewritten query
    2. Mechanism / definition perspective
    3. Practical implementation / cause perspective
    """
    base_rewrite = rewrite_query(query)
    expanded = expand_query(base_rewrite)

    subqueries = [base_rewrite]

    # Perspective 2: Mechanistic / theoretical
    mech = f"{base_rewrite} mechanism principles architecture"
    subqueries.append(mech)

    # Perspective 3: Expanded / practical terms
    if expanded != base_rewrite:
        subqueries.append(expanded)
    else:
        subqueries.append(f"{base_rewrite} implementation troubleshooting examples")

    # Perspective 4 if requested
    if count >= 4:
        subqueries.append(f"{base_rewrite} optimization best practices")

    # Deduplicate while preserving order
    seen = set()
    result = []
    for sq in subqueries:
        if sq not in seen:
            seen.add(sq)
            result.append(sq)

    return result[:count]


class QueryOptimizer:
    """
    Orchestrates query expansion, query rewriting, and multi-query retrieval.
    """
    def __init__(
        self,
        enable_expansion: bool = True,
        enable_rewriting: bool = True,
        enable_multi_query: bool = False,
        multi_query_count: int = 3
    ):
        self.enable_expansion = enable_expansion
        self.enable_rewriting = enable_rewriting
        self.enable_multi_query = enable_multi_query
        self.multi_query_count = multi_query_count

    def process_query(self, query: str) -> str:
        """Applies configured rewriting and expansion pipeline to a single query."""
        res = query
        if self.enable_rewriting:
            res = rewrite_query(res)
        if self.enable_expansion:
            res = expand_query(res)
        return res

    def multi_query_search(
        self,
        search_fn: Callable[[str, int], List[ScoredChunk]],
        query: str,
        top_k: int = 5,
        k0: int = 60
    ) -> List[ScoredChunk]:
        """
        Executes multi-query retrieval:
        1. Deconstructs query into multiple subqueries.
        2. Retrieves candidate lists for each subquery.
        3. Merges all candidate lists using Reciprocal Rank Fusion (RRF).
        """
        subqueries = generate_subqueries(query, count=self.multi_query_count)
        ranked_lists: List[List[ScoredChunk]] = []

        for sq in subqueries:
            results = search_fn(sq, top_k * 2)
            ranked_lists.append(results)

        return reciprocal_rank_fusion(ranked_lists, k0=k0, top_k=top_k)
