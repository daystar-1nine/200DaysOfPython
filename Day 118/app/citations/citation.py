"""
Citation generation, formatting, and attribution verification for Advanced RAG.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from app.index.vector_index import ScoredChunk


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
        """Formats standard citation tag: [Source: Title - Chunk Index]."""
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


def extract_citations_from_scored_chunks(
    scored_chunks: List[ScoredChunk],
    max_snippet_chars: int = 120
) -> List[Citation]:
    """Extracts citation objects from ScoredChunk records."""
    citations = []
    for sc in scored_chunks:
        meta = sc.chunk.metadata
        text_clean = sc.chunk.text.strip()
        snippet = (text_clean[:max_snippet_chars] + "...") if len(text_clean) > max_snippet_chars else text_clean
        citations.append(Citation(
            source_id=meta.chunk_id,
            document_id=meta.document_id,
            title=meta.title,
            category=meta.category,
            chunk_index=meta.chunk_index,
            snippet=snippet,
            relevance_score=round(float(sc.score), 4)
        ))
    return citations


def format_citations_block(citations: List[Citation]) -> str:
    """Formats list of citations as a markdown sources block."""
    if not citations:
        return ""
    lines = ["\n**Sources & Citations:**"]
    for idx, c in enumerate(citations, start=1):
        lines.append(f"{idx}. {c.format_citation_tag()} (Score: {c.relevance_score:.4f}): \"{c.snippet}\"")
    return "\n".join(lines)
