"""
Document loader supporting text, markdown, CSV, and JSON files.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
import json
import re

from app.ingestion.metadata import DocumentMetadata
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
    for cat in ["python", "data_science", "machine_learning", "deep_learning", "dbms", "nlp", "computer_vision"]:
        if stem.startswith(cat) or cat in stem:
            return cat
    return "general"


def load_document(file_path: Path) -> Document:
    """
    Loads and cleans a single document file.
    Supports .txt, .md, .csv, and .json files.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    ext = file_path.suffix.lower()
    content = ""

    if ext in {".txt", ".md"}:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        content = clean_text(raw_text)

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # If JSON has "content" or "text", use that; otherwise format entries
            if "content" in data:
                content = clean_text(str(data["content"]))
            elif "text" in data:
                content = clean_text(str(data["text"]))
            else:
                content = clean_text("\n".join(f"{k}: {v}" for k, v in data.items()))
        elif isinstance(data, list):
            content = clean_text("\n".join(str(item) for item in data))

    elif ext == ".csv":
        rows = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        content = clean_text("\n".join(rows))

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    title = _extract_title(content, file_path.stem)
    category = _infer_category(file_path)
    doc_id = f"doc_{file_path.stem.lower()}"
    words = tokenize_words(content)

    metadata = DocumentMetadata(
        document_id=doc_id,
        title=title,
        category=category,
        source_path=str(file_path),
        file_type=ext,
        char_count=len(content),
        word_count=len(words)
    )

    return Document(metadata=metadata, content=content)


def load_directory(dir_path: Path, recursive: bool = True) -> List[Document]:
    """
    Loads all supported documents from a directory.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []

    pattern = "**/*" if recursive else "*"
    supported = {".txt", ".md", ".json", ".csv"}
    documents = []

    for path in sorted(dir_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in supported:
            try:
                doc = load_document(path)
                if doc.content:
                    documents.append(doc)
            except Exception as e:
                # Log or handle corrupted file gracefully
                continue

    return documents
