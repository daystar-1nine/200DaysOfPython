"""
Document loader supporting text, markdown, CSV, and JSON files.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
import json
import re

from app.ingestion.metadata import DocumentMetadata, create_document_id
from app.ingestion.cleaner import clean_text, tokenize_words


@dataclass
class Document:
    """Represents an ingested document with metadata and cleaned content."""
    metadata: DocumentMetadata
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "content": self.content
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            metadata=DocumentMetadata.from_dict(data["metadata"]),
            content=data["content"]
        )


def _extract_title(text: str, filename: str) -> str:
    """Extracts title from top-level markdown header # Title or falls back to filename stem."""
    first_lines = text.split("\n")[:5]
    for line in first_lines:
        line_clean = line.strip()
        if line_clean.startswith("# "):
            return line_clean[2:].strip()
    return filename.replace("_", " ").replace("-", " ").title()


def _infer_category(path: Path) -> str:
    """Infers category from parent directory name or filename prefix."""
    parent = path.parent.name.lower()
    if parent not in {"raw", "data", "processed", "input", "documents", "."}:
        return parent
    
    stem = path.stem.lower()
    for cat in ["python", "data_science", "machine_learning", "deep_learning", "dbms", "nlp", "computer_vision", "systems", "networking"]:
        if stem.startswith(cat) or cat in stem:
            return cat
    return "general"


def _infer_tags(path: Path, content: str) -> List[str]:
    """Infers domain tags from content keywords and file name."""
    tags = set()
    category = _infer_category(path)
    tags.add(category)
    content_lower = content.lower()
    
    tag_keywords = {
        "sql": ["sql", "join", "query", "database", "table", "relational"],
        "python": ["python", "comprehension", "gil", "generator", "decorator"],
        "networking": ["http", "tcp", "ip", "dns", "rest", "status code", "404"],
        "machine_learning": ["overfitting", "regularization", "gradient descent", "loss", "model", "epoch"],
        "nlp": ["tf-idf", "bm25", "embedding", "token", "vector", "language"],
        "deep_learning": ["neural", "transformer", "attention", "backpropagation"]
    }
    for tag, kws in tag_keywords.items():
        if any(kw in content_lower for kw in kws):
            tags.add(tag)
            
    return sorted(list(tags))


def load_document(file_path: Path) -> Document:
    """Loads and cleans a single document file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    ext = path.suffix.lower()
    raw_text = ""
    extra: Dict[str, Any] = {}

    if ext in {".txt", ".md"}:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    elif ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            raw_text = data.get("content", data.get("text", json.dumps(data, indent=2)))
            extra = {k: v for k, v in data.items() if k not in {"content", "text"}}
        elif isinstance(data, list):
            raw_text = "\n\n".join(str(item) for item in data)
    elif ext == ".csv":
        rows = []
        with open(path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))
        raw_text = "\n".join(rows)
    else:
        raise ValueError(f"Unsupported document format: {ext}")

    cleaned = clean_text(raw_text)
    title = _extract_title(cleaned, path.stem)
    category = _infer_category(path)
    tags = _infer_tags(path, cleaned)

    tokens = tokenize_words(cleaned)
    meta = DocumentMetadata(
        document_id=create_document_id(path),
        title=title,
        category=category,
        source_path=str(path.resolve()),
        file_type=ext.lstrip("."),
        char_count=len(cleaned),
        word_count=len(tokens),
        tags=tags,
        extra=extra
    )
    return Document(metadata=meta, content=cleaned)


def load_directory(
    dir_path: Path,
    supported_extensions: Optional[List[str]] = None
) -> List[Document]:
    """Recursively loads all matching documents from a directory."""
    dir_p = Path(dir_path)
    if not dir_p.is_dir():
        return []

    exts = tuple(supported_extensions or [".txt", ".md", ".json", ".csv"])
    docs: List[Document] = []

    for file_path in sorted(dir_p.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in exts:
            try:
                docs.append(load_document(file_path))
            except Exception as e:
                print(f"[Warning] Failed to load {file_path}: {e}")

    return docs
