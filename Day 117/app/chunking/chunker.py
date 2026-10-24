"""
Text chunking engine with character, word, and sentence-aware boundary preservation.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re

from app.ingestion.metadata import ChunkMetadata
from app.ingestion.loader import Document
from app.ingestion.cleaner import tokenize_words


@dataclass
class Chunk:
    """Represents an indexed chunk with textual content and associated metadata."""
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            text=data["text"],
            metadata=ChunkMetadata.from_dict(data["metadata"])
        )


class TextChunker:
    """
    Configurable document chunker supporting character, word, and sentence-level partitioning.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 50, strategy: str = "sentence"):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap cannot be negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy.lower()

    def chunk_document(
        self,
        doc: Document,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Chunk]:
        """Splits an ingested document into chunks inheriting document metadata."""
        return self.chunk_text(
            text=doc.content,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc.metadata.document_id,
            title=doc.metadata.title,
            category=doc.metadata.category,
            source_file=doc.metadata.source_path
        )

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        doc_id: str = "doc_0",
        title: str = "",
        category: str = "",
        source_file: str = ""
    ) -> List[Chunk]:
        """
        Splits raw text into chunks using configured strategy, preserving boundaries and metadata.
        """
        c_size = chunk_size if chunk_size is not None else self.chunk_size
        ov = overlap if overlap is not None else self.overlap

        if c_size <= 0 or ov < 0 or ov >= c_size:
            raise ValueError(f"Invalid chunk_size ({c_size}) or overlap ({ov})")

        if not text or not text.strip():
            return []

        clean_str = text.strip()

        if self.strategy == "character":
            chunk_texts, spans = self._chunk_characters(clean_str, c_size, ov)
        elif self.strategy == "word":
            chunk_texts, spans = self._chunk_words(clean_str, c_size, ov)
        else:  # "sentence"
            chunk_texts, spans = self._chunk_sentences(clean_str, c_size, ov)

        chunks: List[Chunk] = []
        for idx, (t, (start_c, end_c)) in enumerate(zip(chunk_texts, spans)):
            words = tokenize_words(t)
            meta = ChunkMetadata(
                chunk_id=f"{doc_id}_c{idx}",
                document_id=doc_id,
                chunk_index=idx,
                title=title,
                category=category,
                start_char=start_c,
                end_char=end_c,
                token_count=len(words),
                word_count=len(words),
                source_file=source_file
            )
            chunks.append(Chunk(text=t, metadata=meta))

        return chunks

    def _chunk_characters(self, text: str, chunk_size: int, overlap: int):
        stride = chunk_size - overlap
        chunks, spans = [], []
        n = len(text)
        start = 0

        while start < n:
            end = min(start + chunk_size, n)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
                spans.append((start, end))
            if end == n:
                break
            start += stride

        return chunks, spans

    def _chunk_words(self, text: str, chunk_size: int, overlap: int):
        words = text.split()
        stride = chunk_size - overlap
        chunks, spans = [], []
        n = len(words)
        start = 0

        # Build rough character index mapping
        char_offsets = []
        curr = 0
        for w in words:
            found = text.find(w, curr)
            if found != -1:
                char_offsets.append((found, found + len(w)))
                curr = found + len(w)
            else:
                char_offsets.append((curr, curr + len(w)))
                curr += len(w) + 1

        while start < n:
            end = min(start + chunk_size, n)
            chunk_str = " ".join(words[start:end])
            start_char = char_offsets[start][0]
            end_char = char_offsets[end - 1][1]
            chunks.append(chunk_str)
            spans.append((start_char, end_char))
            if end == n:
                break
            start += stride

        return chunks, spans

    def _chunk_sentences(self, text: str, chunk_size: int, overlap: int):
        """
        Splits by sentences and bundles them until chunk_size (in characters) is approached,
        preserving sentence integrity.
        """
        # Split on sentence terminals while keeping punctuation
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            # Fall back to character chunker if no distinct sentences
            return self._chunk_characters(text, chunk_size, overlap)

        chunks, spans = [], []

        # Build sentence character spans
        s_spans = []
        curr_idx = 0
        for s in sentences:
            found = text.find(s, curr_idx)
            if found != -1:
                s_spans.append((found, found + len(s)))
                curr_idx = found + len(s)
            else:
                s_spans.append((curr_idx, curr_idx + len(s)))
                curr_idx += len(s) + 1

        i = 0
        while i < len(sentences):
            s = sentences[i]
            s_len = len(s)

            # If a single sentence is larger than chunk_size, character-chunk it directly
            if s_len >= chunk_size:
                sub_chunks, sub_spans = self._chunk_characters(s, chunk_size, overlap)
                offset = s_spans[i][0]
                for sc, (s_start, s_end) in zip(sub_chunks, sub_spans):
                    chunks.append(sc)
                    spans.append((offset + s_start, offset + s_end))
                i += 1
                continue

            # Build window
            window = [s]
            curr_char_count = s_len
            start_s_idx = i
            j = i + 1

            while j < len(sentences):
                next_s = sentences[j]
                next_len = len(next_s)
                if curr_char_count + next_len + 1 > chunk_size:
                    break
                window.append(next_s)
                curr_char_count += next_len + 1
                j += 1

            chunk_str = " ".join(window)
            start_char = s_spans[start_s_idx][0]
            end_char = s_spans[j - 1][1]
            chunks.append(chunk_str)
            spans.append((start_char, end_char))

            if j >= len(sentences):
                break

            # Calculate advance step based on overlap in characters
            ov_chars = 0
            backtrack = j
            while backtrack > start_s_idx + 1 and ov_chars < overlap:
                backtrack -= 1
                ov_chars += len(sentences[backtrack]) + 1

            i = max(start_s_idx + 1, backtrack)

        return chunks, spans


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: Optional[int] = None,
    strategy: str = "sentence"
) -> List[Chunk]:
    """Convenience function to chunk text directly."""
    if overlap is None:
        overlap = min(50, max(0, chunk_size // 5))
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap, strategy=strategy)
    return chunker.chunk_text(text)
