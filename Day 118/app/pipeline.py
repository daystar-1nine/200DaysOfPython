"""
End-to-End Advanced RAG Pipeline Orchestrator.
Integrates Ingestion, Chunking, Embeddings, BM25, Dense Retrieval, RRF Fusion,
Query Optimization, Cross-Encoder Reranking, MMR Diversity, Context Compression, and Grounded Generation.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import time

from app.config import AdvancedRAGConfig
from app.ingestion.loader import Document, load_directory, load_document
from app.chunking.chunker import TextChunker, Chunk
from app.embeddings.embedder import DenseTFIDFEmbedder, get_embedder
from app.index.vector_index import VectorIndex, ScoredChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.query_expansion import QueryOptimizer, rewrite_query, expand_query
from app.retrieval.reranker import CrossEncoderReranker, IdentityReranker
from app.retrieval.mmr import MaximalMarginalRelevance
from app.retrieval.metadata_filter import MetadataFilter
from app.context.compressor import ContextCompressor, CompressedContext
from app.generation.prompt import AdvancedRAGPromptBuilder
from app.generation.generator import AdvancedRAGGenerator, RAGResponse


class AdvancedRAGPipeline:
    """
    Modular, state-of-the-art Advanced RAG pipeline.
    """
    def __init__(self, config: Optional[AdvancedRAGConfig] = None):
        self.config = config or AdvancedRAGConfig()
        
        # Modules
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy
        )
        self.embedder = get_embedder(
            name=self.config.embedding_type,
            dim=self.config.embedding_dim
        )
        self.vector_index = VectorIndex(dimension=self.config.embedding_dim)
        self.bm25_retriever = BM25Retriever(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            epsilon=self.config.bm25_epsilon
        )
        self.dense_retriever = DenseRetriever(
            embedder=self.embedder,
            index=self.vector_index
        )
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            alpha=self.config.hybrid_alpha,
            candidate_k=self.config.candidate_k,
            rrf_k0=self.config.rrf_k0
        )
        self.query_optimizer = QueryOptimizer(
            enable_expansion=self.config.enable_query_expansion,
            enable_rewriting=self.config.enable_query_rewriting,
            enable_multi_query=self.config.enable_multi_query,
            multi_query_count=self.config.multi_query_count
        )
        self.reranker = CrossEncoderReranker()
        self.mmr_selector = MaximalMarginalRelevance(lambda_mult=self.config.mmr_lambda)
        self.compressor = ContextCompressor(
            min_sentence_score=self.config.compression_similarity_threshold,
            max_context_length=self.config.max_context_length
        )
        self.generator = AdvancedRAGGenerator(
            prompt_builder=AdvancedRAGPromptBuilder(),
            min_relevance_threshold=self.config.min_score_threshold,
            include_citations=self.config.include_citations
        )
        
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self._is_indexed: bool = False

    def ingest_documents(self, documents: List[Document]) -> "AdvancedRAGPipeline":
        """Ingests, chunks, embeds, and indexes a collection of Documents."""
        if not documents:
            return self

        self.documents = list(documents)
        all_chunks: List[Chunk] = []
        for doc in self.documents:
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        self.chunks = all_chunks
        corpus_texts = [c.text for c in self.chunks]

        # 1. Fit embedder and populate VectorIndex
        self.embedder.fit(corpus_texts)
        embeddings = self.embedder.embed_batch(corpus_texts)
        self.vector_index.clear()
        self.vector_index.add_chunks(self.chunks, embeddings)

        # 2. Index BM25
        self.bm25_retriever.index(self.chunks)

        self._is_indexed = True
        return self

    def ingest_directory(self, dir_path: Path) -> "AdvancedRAGPipeline":
        """Loads and ingests all documents from directory."""
        docs = load_directory(dir_path, supported_extensions=list(self.config.supported_extensions))
        return self.ingest_documents(docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid_rrf",
        enable_rewriting: bool = False,
        enable_expansion: bool = False,
        enable_rerank: bool = False,
        enable_mmr: bool = False,
        mmr_lambda: Optional[float] = None,
        filter_fn: Optional[Callable[[Chunk], bool]] = None
    ) -> List[ScoredChunk]:
        """
        Executes multi-stage retrieval pipeline with configurable modules.
        """
        if not self._is_indexed or not self.chunks:
            return []

        # Stage 1: Query Optimization
        processed_query = query
        if enable_rewriting:
            processed_query = rewrite_query(processed_query)
        if enable_expansion:
            processed_query = expand_query(processed_query)

        # Stage 2: Initial Candidate Retrieval
        candidate_k = max(top_k * 3, self.config.candidate_k)
        if method == "bm25":
            candidates = self.bm25_retriever.search(processed_query, top_k=candidate_k, filter_fn=filter_fn)
        elif method == "dense":
            candidates = self.dense_retriever.search(processed_query, top_k=candidate_k, filter_fn=filter_fn)
        elif method == "hybrid_weighted":
            candidates = self.hybrid_retriever.search_weighted(processed_query, top_k=candidate_k, filter_fn=filter_fn)
        else:  # "hybrid_rrf"
            candidates = self.hybrid_retriever.search_rrf(processed_query, top_k=candidate_k, filter_fn=filter_fn)

        if not candidates:
            return []

        # Stage 3: Cross-Encoder Reranking
        if enable_rerank:
            candidates = self.reranker.rerank(processed_query, candidates, top_k=candidate_k)

        # Stage 4: MMR Diversification
        if enable_mmr:
            lam = mmr_lambda if mmr_lambda is not None else self.config.mmr_lambda
            candidates = self.mmr_selector.select(candidates, top_k=top_k, lambda_mult=lam)
        else:
            candidates = candidates[:top_k]

        return candidates

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid_rrf",
        enable_rewriting: bool = True,
        enable_expansion: bool = True,
        enable_rerank: bool = True,
        enable_mmr: bool = True,
        enable_compression: bool = True,
        filter_fn: Optional[Callable[[Chunk], bool]] = None
    ) -> RAGResponse:
        """
        End-to-End RAG query execution from question to grounded answer with citations.
        """
        retrieved_chunks = self.retrieve(
            query=query,
            top_k=top_k,
            method=method,
            enable_rewriting=enable_rewriting,
            enable_expansion=enable_expansion,
            enable_rerank=enable_rerank,
            enable_mmr=enable_mmr,
            filter_fn=filter_fn
        )

        if not retrieved_chunks:
            return RAGResponse(
                question=query,
                answer="I don't have enough information in the provided documents to answer this question.",
                citations=[],
                retrieved_chunks=[],
                is_supported=False,
                context_used="",
                compression_ratio=0.0,
                generation_time_ms=0.0
            )

        # Stage 5: Context Compression
        if enable_compression:
            context = self.compressor.compress(query, retrieved_chunks)
        else:
            # Uncompressed raw bundle
            full_text = "\n\n".join(
                f"[Source: {sc.chunk.metadata.document_id} | Chunk: {sc.chunk.metadata.chunk_index}]\n{sc.chunk.text}"
                for sc in retrieved_chunks
            )
            context = CompressedContext(
                text=full_text,
                original_char_count=len(full_text),
                compressed_char_count=len(full_text),
                compression_ratio=0.0,
                retained_sentences_count=0,
                total_sentences_count=0,
                chunks_used=retrieved_chunks
            )

        # Stage 6: Grounded Generation
        return self.generator.generate(query, context)
