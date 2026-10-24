"""
End-to-end RAG Pipeline coordinating ingestion, indexing, retrieval, and generation.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import RAGConfig
from app.ingestion.loader import Document, load_directory, load_document
from app.chunking.chunker import Chunk, TextChunker
from app.embeddings.embedder import BaseEmbedder, DenseTFIDFEmbedder
from app.index.vector_index import VectorIndex
from app.retrieval.keyword import KeywordRetriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.top_k import format_retrieval_results, QueryResult
from app.generation.generator import RAGGenerator, RAGResponse
from app.generation.prompt import RAGPromptBuilder, ContextBuilder


class RAGPipeline:
    """
    Unified RAG Pipeline connecting document ingestion, chunking,
    vector indexing, hybrid retrieval, and grounded answer generation.
    """
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedder: Optional[BaseEmbedder] = None
    ):
        self.config = config or RAGConfig()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy
        )
        self.embedder = embedder or DenseTFIDFEmbedder(dim=self.config.embedding_dim)
        self.index = VectorIndex(dimension=self.config.embedding_dim)

        self.keyword_retriever = KeywordRetriever()
        self.semantic_retriever = SemanticRetriever(self.embedder, self.index)
        self.hybrid_retriever = HybridRetriever(
            self.keyword_retriever,
            self.semantic_retriever,
            alpha=self.config.hybrid_alpha
        )

        self.generator = RAGGenerator(
            prompt_builder=RAGPromptBuilder(
                context_builder=ContextBuilder(max_chars=self.config.max_context_length)
            ),
            min_relevance_threshold=self.config.min_score_threshold,
            include_citations=self.config.include_citations
        )

        self.chunks: List[Chunk] = []
        self._is_ready = False

    def ingest_documents(self, documents: List[Document]) -> int:
        """
        Processes a list of ingested documents: chunks text, computes embeddings,
        and constructs vector and lexical indices.
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            return 0

        self.chunks = all_chunks
        texts = [c.text for c in all_chunks]

        # 1. Fit embedder & generate embeddings
        self.embedder.fit(texts)
        embeddings = self.embedder.embed_batch(texts)

        # 2. Build Vector Index
        self.index.clear()
        self.index.add_chunks(all_chunks, embeddings)

        # 3. Build Keyword Inverted Index
        self.keyword_retriever.index(all_chunks)

        self._is_ready = True
        return len(all_chunks)

    def ingest_directory(self, dir_path: Path) -> int:
        """Loads and indexes all documents from target directory."""
        docs = load_directory(dir_path)
        return self.ingest_documents(docs)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[QueryResult]:
        """
        Retrieves top_k relevant chunks using specified retrieval method:
        'keyword', 'semantic', or 'hybrid'.
        """
        if not self._is_ready or self.index.size() == 0:
            return []

        k = top_k if top_k is not None else self.config.top_k
        meth = (method if method is not None else self.config.retrieval_method).lower()

        if meth == "keyword":
            raw = self.keyword_retriever.retrieve(
                query, top_k=k, category_filter=category_filter
            )
        elif meth == "semantic":
            raw = self.semantic_retriever.retrieve(
                query, top_k=k, category_filter=category_filter
            )
        elif meth == "hybrid":
            raw = self.hybrid_retriever.retrieve(
                query, top_k=k, category_filter=category_filter
            )
        else:
            raise ValueError(f"Unknown retrieval method '{meth}'. Choose 'keyword', 'semantic', 'hybrid'.")

        return format_retrieval_results(raw, method=meth)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        Executes end-to-end RAG query: retrieves top context chunks,
        constructs grounded prompt, and generates answer with citations.
        """
        results = self.retrieve(
            query=question,
            top_k=top_k,
            method=method,
            category_filter=category_filter
        )
        return self.generator.generate(question=question, retrieved_results=results)

    def save_index(self, dir_path: Path) -> None:
        """Saves vector index to disk."""
        self.index.save(dir_path)

    def load_index(self, dir_path: Path) -> None:
        """Loads index from disk and reinitializes retrieval components."""
        self.index.load(dir_path)
        self.chunks = self.index.all_chunks()
        if self.chunks:
            self.keyword_retriever.index(self.chunks)
            texts = [c.text for c in self.chunks]
            self.embedder.fit(texts)
            self._is_ready = True
