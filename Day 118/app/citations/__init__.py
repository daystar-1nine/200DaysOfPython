"""
Citations package for source attribution and verification.
"""
from app.citations.citation import (
    Citation,
    extract_citations_from_scored_chunks,
    format_citations_block,
)

__all__ = [
    "Citation",
    "extract_citations_from_scored_chunks",
    "format_citations_block",
]
