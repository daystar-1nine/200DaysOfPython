"""
Metadata Filtering Engine.
Supports Pre-filtering and Post-filtering on chunk and document attributes,
including category, document IDs, tags, token limits, and logical combinators.
"""
from typing import List, Dict, Any, Optional, Callable, Sequence
from dataclasses import dataclass

from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk


class MetadataFilter:
    """
    Evaluates metadata conditions and filters candidate chunks or scored results.
    """
    @staticmethod
    def build_predicate(
        category: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
        document_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        tag_mode: str = "any",  # "any" or "all"
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        custom_fn: Optional[Callable[[Chunk], bool]] = None
    ) -> Callable[[Chunk], bool]:
        """
        Constructs a composable filter predicate function (Chunk -> bool).
        """
        allowed_cats = set([c.lower() for c in categories]) if categories else None
        if category:
            allowed_cats = {category.lower()}

        allowed_docs = set(document_ids) if document_ids else None
        if document_id:
            allowed_docs = {document_id}

        required_tags = [t.lower() for t in tags] if tags else []

        def predicate(chunk: Chunk) -> bool:
            meta = chunk.metadata

            # Category filter
            if allowed_cats is not None:
                if meta.category.lower() not in allowed_cats:
                    return False

            # Document ID filter
            if allowed_docs is not None:
                if meta.document_id not in allowed_docs:
                    return False

            # Tags filter
            if required_tags:
                chunk_tags = set(t.lower() for t in getattr(meta, "tags", []))
                if tag_mode == "all":
                    if not all(t in chunk_tags for t in required_tags):
                        return False
                else:  # "any"
                    if not any(t in chunk_tags for t in required_tags):
                        return False

            # Token range
            if min_tokens is not None and meta.token_count < min_tokens:
                return False
            if max_tokens is not None and meta.token_count > max_tokens:
                return False

            # Custom user predicate
            if custom_fn is not None and not custom_fn(chunk):
                return False

            return True

        return predicate

    @classmethod
    def pre_filter(
        cls,
        chunks: List[Chunk],
        predicate: Callable[[Chunk], bool]
    ) -> List[Chunk]:
        """
        Pre-filtering: narrows down the corpus of chunks BEFORE retrieval or embedding search.
        Reduces search space, boosts latency, and guarantees strict constraint satisfaction.
        """
        return [c for c in chunks if predicate(c)]

    @classmethod
    def post_filter(
        cls,
        scored_chunks: List[ScoredChunk],
        predicate: Callable[[Chunk], bool]
    ) -> List[ScoredChunk]:
        """
        Post-filtering: filters candidates AFTER initial retrieval.
        Useful when dense/BM25 indices do not natively support inverted-index pre-filtering.
        """
        return [sc for sc in scored_chunks if predicate(sc.chunk)]

    @classmethod
    def and_filters(
        cls,
        *predicates: Callable[[Chunk], bool]
    ) -> Callable[[Chunk], bool]:
        """Combines multiple predicates with logical AND."""
        return lambda c: all(p(c) for p in predicates)

    @classmethod
    def or_filters(
        cls,
        *predicates: Callable[[Chunk], bool]
    ) -> Callable[[Chunk], bool]:
        """Combines multiple predicates with logical OR."""
        return lambda c: any(p(c) for p in predicates)

    @classmethod
    def not_filter(
        cls,
        predicate: Callable[[Chunk], bool]
    ) -> Callable[[Chunk], bool]:
        """Inverts a predicate with logical NOT."""
        return lambda c: not predicate(c)
