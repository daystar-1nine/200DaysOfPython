"""
Vector index for fast similarity search with metadata storage and persistence.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, NamedTuple, Callable
import json
import numpy as np

from app.chunking.chunker import Chunk
from app.embeddings.similarity import batch_cosine_similarity


class ScoredChunk(NamedTuple):
    chunk: Chunk
    score: float


class SizeInt(int):
    """Integer that can also be called as a function for backwards compatibility."""
    def __call__(self) -> int:
        return int(self)


class VectorIndex:
    """
    In-memory vector store mapping chunks to dense embedding vectors.
    Supports top-K cosine similarity search, metadata filtering, and disk serialization.
    """
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._chunk_map: Dict[str, Chunk] = {}

    @property
    def size(self) -> SizeInt:
        """Returns total number of indexed chunks."""
        return SizeInt(len(self.chunks))

    def __len__(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        """Empties the index."""
        self.chunks = []
        self.embeddings = None
        self._chunk_map = {}

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Looks up a chunk by its unique ID."""
        return self._chunk_map.get(chunk_id)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Alias for get_chunk_by_id."""
        return self.get_chunk_by_id(chunk_id)

    def all_chunks(self) -> List[Chunk]:
        """Returns all indexed chunks."""
        return list(self.chunks)

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Adds text chunks with precomputed embedding vectors to the index."""
        if not chunks:
            return

        emb = np.asarray(embeddings, dtype=np.float64)
        if len(chunks) != emb.shape[0]:
            raise ValueError(f"Chunk count {len(chunks)} does not match embeddings rows {emb.shape[0]}")

        if emb.shape[1] != self.dimension:
            raise ValueError(f"Embedding dim {emb.shape[1]} does not match index dim {self.dimension}")

        for c in chunks:
            self._chunk_map[c.metadata.chunk_id] = c

        if self.embeddings is None or len(self.chunks) == 0:
            self.chunks = list(chunks)
            self.embeddings = emb
        else:
            self.chunks.extend(chunks)
            self.embeddings = np.vstack([self.embeddings, emb])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        category: Optional[str] = None,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
        min_score: float = -1.0
    ) -> List[ScoredChunk]:
        """Searches index using cosine similarity with optional metadata filtering."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Vectorized cosine similarity
        sims = batch_cosine_similarity(query_vector, self.embeddings)

        scored: List[ScoredChunk] = []
        for idx, (sim, chunk) in enumerate(zip(sims, self.chunks)):
            # Category filter
            if category is not None and chunk.metadata.category.lower() != category.lower():
                continue
            # Generic predicate filter
            if filter_fn is not None and not filter_fn(chunk):
                continue
            # Score threshold
            if sim < min_score:
                continue

            scored.append(ScoredChunk(chunk=chunk, score=float(sim)))

        # Sort descending by similarity score
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def save(self, directory: Path) -> None:
        """Serializes chunks and embeddings matrix to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        chunks_data = [c.to_dict() for c in self.chunks]
        (target_dir / "chunks.json").write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

        meta = {
            "dimension": self.dimension,
            "chunk_count": len(self.chunks)
        }
        (target_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if self.embeddings is not None:
            np.save(str(target_dir / "embeddings.npy"), self.embeddings)

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        """Loads serialized index from disk."""
        target_dir = Path(directory)
        meta = json.loads((target_dir / "index_meta.json").read_text(encoding="utf-8"))
        index = cls(dimension=meta["dimension"])

        chunks_data = json.loads((target_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(d) for d in chunks_data]

        emb_file = target_dir / "embeddings.npy"
        if emb_file.exists():
            embeddings = np.load(str(emb_file))
            index.add_chunks(chunks, embeddings)
        else:
            index.chunks = chunks
            for c in chunks:
                index._chunk_map[c.metadata.chunk_id] = c

        return index
