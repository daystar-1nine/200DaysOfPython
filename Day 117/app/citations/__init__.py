"""
Citation generation and source attribution package.
"""
from app.citations.citation import (
    Citation,
    extract_citations_from_results,
    format_citations_block,
    verify_citation_coverage
)

__all__ = [
    "Citation",
    "extract_citations_from_results",
    "format_citations_block",
    "verify_citation_coverage"
]
