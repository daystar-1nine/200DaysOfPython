"""
Metadata models for documents and chunks.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import time


def create_document_id(path: Path) -> str:
    """Creates a deterministic document ID string from a file path."""
    return f"doc_{path.stem.lower()}"


@dataclass
class DocumentMetadata:
    """Metadata associated with an ingested source document."""
    document_id: str
    title: str
    category: str
    source_path: str
    file_type: str
    char_count: int = 0
    word_count: int = 0
    created_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        return cls(**data)


@dataclass
class ChunkMetadata:
    """Metadata associated with an individual text chunk."""
    chunk_id: str
    document_id: str
    chunk_index: int
    title: str
    category: str
    start_char: int
    end_char: int
    token_count: int
    word_count: int
    source_file: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        return cls(**data)
