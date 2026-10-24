"""
Vector index for fast similarity search with metadata storage and persistence.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np

from app.chunking.chunker import Chunk
from app.embeddings.similarity import batch_cosine_similarity


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

    def size(self) -> int:
        """Returns total number of indexed chunks."""
        return len(self.chunks)

    def clear(self) -> None:
        """Empties the index."""
        self.chunks = []
        self.embeddings = None
        self._chunk_map = {}

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Looks up a chunk by its unique ID."""
        return self._chunk_map.get(chunk_id)

    def all_chunks(self) -> List[Chunk]:
        """Returns all indexed chunks."""
        return list(self.chunks)

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Adds new chunks and their corresponding embedding vectors to the index.
        """
        if not chunks:
            return

        emb_mat = np.asarray(embeddings, dtype=np.float64)
        if emb_mat.ndim == 1:
            emb_mat = emb_mat.reshape(1, -1)

        if len(chunks) != emb_mat.shape[0]:
            raise ValueError(f"Chunk count ({len(chunks)}) must match embedding row count ({emb_mat.shape[0]})")

        if emb_mat.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {emb_mat.shape[1]} does not match index dimension {self.dimension}")

        for c in chunks:
            self._chunk_map[c.metadata.chunk_id] = c

        if self.embeddings is None or self.embeddings.shape[0] == 0:
            self.embeddings = emb_mat
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, emb_mat])
            self.chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        category_filter: Optional[str] = None,
        doc_id_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Searches the index for the top_k most similar chunks.
        Filters by category or document_id if specified.
        Returns List of (Chunk, similarity_score) sorted descending by score.
        """
        if self.size() == 0 or self.embeddings is None:
            return []

        sims = batch_cosine_similarity(query_vector, self.embeddings)

        results: List[Tuple[Chunk, float]] = []
        for idx, score in enumerate(sims):
            if score < min_score:
                continue

            chunk = self.chunks[idx]
            meta = chunk.metadata

            if category_filter and meta.category.lower() != category_filter.lower():
                continue

            if doc_id_filter and meta.document_id.lower() != doc_id_filter.lower():
                continue

            results.append((chunk, float(score)))

        # Sort descending by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, dir_path: Path) -> None:
        """Persists vector array to .npz and chunk metadata to JSON."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        vec_file = dir_path / "vectors.npz"
        emb_data = self.embeddings if self.embeddings is not None else np.empty((0, self.dimension))
        np.savez_compressed(vec_file, embeddings=emb_data)

        # Save chunk records
        meta_file = dir_path / "metadata.json"
        chunk_data = [c.to_dict() for c in self.chunks]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "count": len(self.chunks),
                "chunks": chunk_data
            }, f, indent=2)

    def load(self, dir_path: Path) -> None:
        """Loads index from persisted directory files."""
        dir_path = Path(dir_path)
        vec_file = dir_path / "vectors.npz"
        meta_file = dir_path / "metadata.json"

        if not vec_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index files missing in {dir_path}")

        # Load embeddings
        with np.load(vec_file) as data:
            self.embeddings = data["embeddings"]

        # Load chunk records
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dimension = data.get("dimension", self.embeddings.shape[1] if self.embeddings.ndim > 1 else 64)
        self.chunks = [Chunk.from_dict(item) for item in data.get("chunks", [])]
        self._chunk_map = {c.metadata.chunk_id: c for c in self.chunks}
