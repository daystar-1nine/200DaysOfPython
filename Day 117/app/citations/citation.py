"""
Citation generation, formatting, and attribution verification.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from app.retrieval.top_k import QueryResult


@dataclass
class Citation:
    """Represents a cited source chunk attributed to a generated statement."""
    source_id: str
    document_id: str
    title: str
    category: str
    chunk_index: int
    snippet: str
    relevance_score: float

    def format_citation_tag(self) -> str:
        """Formats standard human-readable citation tag: [Source: Title - Chunk Index]."""
        clean_title = self.title if self.title else self.document_id
        return f"[Source: {clean_title} - Chunk {self.chunk_index}]"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["citation_tag"] = self.format_citation_tag()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        clean = {k: v for k, v in data.items() if k != "citation_tag"}
        return cls(**clean)


def extract_citations_from_results(
    results: List[QueryResult],
    max_snippet_chars: int = 120
) -> List[Citation]:
    """Extracts citation objects from retrieved QueryResult records."""
    citations = []
    for r in results:
        meta = r.chunk.metadata
        text_clean = r.chunk.text.strip()
        snippet = (text_clean[:max_snippet_chars] + "...") if len(text_clean) > max_snippet_chars else text_clean
        citations.append(Citation(
            source_id=meta.chunk_id,
            document_id=meta.document_id,
            title=meta.title,
            category=meta.category,
            chunk_index=meta.chunk_index,
            snippet=snippet,
            relevance_score=round(r.score, 4)
        ))
    return citations


def format_citations_block(citations: List[Citation]) -> str:
    """Formats list of citations as a cleanly formatted markdown sources block."""
    if not citations:
        return ""
    lines = ["\nSources:"]
    for c in citations:
        lines.append(f"- {c.format_citation_tag()} (Relevance: {c.relevance_score:.2f})")
    return "\n".join(lines)


def verify_citation_coverage(answer: str, citations: List[Citation]) -> Dict[str, Any]:
    """
    Checks if citations are present in the answer text or appended.
    Returns coverage statistics.
    """
    if not citations:
        return {"has_citations": False, "cited_count": 0, "total_sources": 0, "coverage_rate": 0.0}

    found = 0
    missing = []
    for c in citations:
        tag = c.format_citation_tag().lower()
        cid = c.source_id.lower()
        c_title = c.title.lower() if c.title else ""
        ans_lower = answer.lower()
        if tag in ans_lower or cid in ans_lower or (c_title and c_title in ans_lower):
            found += 1
        else:
            missing.append(c.format_citation_tag())

    rate = found / len(citations)
    return {
        "has_citations": found > 0,
        "cited_count": found,
        "total_sources": len(citations),
        "coverage_rate": round(rate, 4),
        "missing_tags": missing
    }
