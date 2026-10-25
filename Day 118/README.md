# 🚀 Day 118: Advanced RAG — Hybrid Search, Reranking & Query Optimization

**Progress:** Day 118 / 200 (59.0% Complete — 82 Days Remaining)  
**Core Mission:** Upgrade baseline RAG into an enterprise-grade retrieval engine combining BM25 keyword matching, dense vector search, Reciprocal Rank Fusion (RRF), query optimization, cross-encoder reranking, Maximal Marginal Relevance (MMR), and context compression.

> *"A more powerful LLM cannot reliably compensate for poor retrieval. The retrieval engine is the foundation of truth in RAG."*

---

## 🏗️ 1. Advanced RAG Architecture

```text
                                ┌───────────────────────────┐
                                │   Raw Documents (.md)     │
                                └─────────────┬─────────────┘
                                              ↓
                                   Ingestion & Cleaning
                                              ↓
                                Sentence-Aware Chunking (400 chars, 50 overlap)
                                              ↓
                   ┌──────────────────────────┴──────────────────────────┐
                   ↓                                                     ↓
       Okapi BM25 Indexing                                   Dense TF-IDF + SVD Index
     (Sparse Inverted Index)                               (VectorIndex Metric Space)
                   │                                                     │
User Query ────────┴─────────────────────────────────────────────────────┘
    │
    ├─► Query Optimization
    │     ├─► Conversational Prefix Rewriting
    │     ├─► Technical Synonym Expansion
    │     └─► Multi-Query Perspective Generation
    │
    ├─► Hybrid Candidate Retrieval (Candidate Pool K=20)
    │     ├─► BM25 Sparse Search (Exact Terms, Codes, IDs)
    │     ├─► Dense Semantic Search (Conceptual Proximity)
    │     └─► Reciprocal Rank Fusion (RRF, k0=60)
    │
    ├─► Metadata Pre/Post-Filtering (Domain Constraints)
    │
    ├─► Stage 2: Cross-Encoder Joint Reranking
    │     └─► Cross-Attention Interaction (Coverage, N-grams, Proximity)
    │
    ├─► Diversity Selection: Maximal Marginal Relevance (MMR, λ=0.7)
    │     └─► Pareto Optimal Relevance vs Redundancy Pruning
    │
    ├─► Extractive Context Compression
    │     └─► Sentence-level Evidence Filtering (-34% token load)
    │
    └─► Grounded Generation & Citation Attribution
          └─► Strict Anti-Hallucination Guardrails + Bracketed Sources
```

---

## 📁 2. Project Directory Structure

```text
Day 118/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Centralized RAG configuration parameters
│   ├── main.py                   # Command-line query interface
│   ├── pipeline.py               # Orchestrator combining all retrieval & generation stages
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── cleaner.py            # Text normalization, tokenization, sentence extraction
│   │   ├── loader.py             # Multi-format document loader (.md, .txt, .json, .csv)
│   │   └── metadata.py           # DocumentMetadata and ChunkMetadata dataclasses
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py            # TextChunker with sentence, word, and character boundaries
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedder.py           # DenseTFIDFEmbedder, SparseTFIDFEmbedder
│   │   └── similarity.py         # Cosine similarity, batch vectorization, Euclidean metric
│   ├── index/
│   │   ├── __init__.py
│   │   └── vector_index.py       # VectorIndex store with disk serialization
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py     # Robertson-Spärck Jones Okapi BM25 implementation
│   │   ├── dense_retriever.py    # Dense semantic embedding retriever
│   │   ├── hybrid.py             # Weighted score fusion & RRF hybrid search
│   │   ├── rrf.py                # Dedicated Reciprocal Rank Fusion algorithm (k0=60)
│   │   ├── query_expansion.py    # Synonym expansion, query rewriting, multi-query
│   │   ├── reranker.py           # CrossEncoderReranker with token cross-attention
│   │   ├── mmr.py                # Maximal Marginal Relevance diversity selector
│   │   └── metadata_filter.py    # Pre/Post filtering predicates (AND/OR/NOT)
│   ├── context/
│   │   ├── __init__.py
│   │   └── compressor.py         # Extractive sentence-level context compressor
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py          # Grounded generator with anti-hallucination guardrails
│   │   └── prompt.py             # Structured RAG prompt builder
│   ├── citations/
│   │   ├── __init__.py
│   │   └── citation.py           # Citation extraction, formatting, and validation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py  # Precision@K, Recall@K, Hit@K, MRR, nDCG@K
│   │   ├── latency.py            # LatencyTracker (P50, P90, P95, mean)
│   │   ├── error_analysis.py     # Error taxonomy & diagnostic classification
│   │   ├── grounding.py          # Faithfulness and citation precision evaluation
│   │   └── evaluator.py          # Benchmark harness runner
│   └── reporting/
│       ├── __init__.py
│       └── charts.py             # 15 publication-grade visualization generators
├── data/
│   ├── generate_knowledge_base.py# 30 technical documents & 60 benchmark queries generator
│   ├── raw/                      # 30 markdown knowledge base documents
│   └── evaluation/
│       └── benchmark_queries.json# 60 benchmark queries with graded relevance labels
├── experiments/
│   └── run_experiments.py        # Complete benchmark runner for all 7 experiments
├── outputs/
│   ├── metrics.csv               # Model-level aggregated IR metrics
│   ├── retrieval_results.csv     # Per-query evaluation records
│   ├── error_analysis.csv        # Failure mode diagnoses
│   └── charts/                   # 15 publication PNG charts
├── scratch/
│   ├── scratch_rrf.py            # Challenge 1: Pure RRF implementation
│   ├── scratch_hybrid.py         # Challenge 2: Min-Max score normalization hybrid
│   ├── scratch_expansion.py      # Challenge 3: Query expansion & rewriting
│   ├── scratch_mmr.py            # Challenge 4: MMR diversity selection
│   ├── scratch_metadata.py       # Challenge 5: Metadata filtering predicates
│   ├── scratch_compressor.py     # Challenge 6: Context compression
│   └── scratch_benchmark.py      # Challenge 7: IR micro-benchmark
├── tests/
│   ├── conftest.py               # Shared pytest fixtures
│   ├── test_bm25_dense.py        # 14 tests for BM25 and Dense engines
│   ├── test_rrf_hybrid.py        # 14 tests for RRF and Hybrid fusion
│   ├── test_query_expansion.py   # 11 tests for query optimization
│   ├── test_reranker.py          # 8 tests for Cross-Encoder reranker
│   ├── test_mmr.py               # 7 tests for MMR diversity
│   ├── test_metadata_filter.py   # 11 tests for metadata filtering
│   ├── test_compressor.py        # 5 tests for context compression
│   ├── test_metrics_ndcg.py      # 10 tests for IR metrics and nDCG
│   ├── test_pipeline_e2e.py      # 7 tests for end-to-end grounded generation
│   └── test_advanced_rag_features.py # 14 tests for supplementary features
├── DAY_118_REPORT.md             # Empirical research report with benchmark data
├── README.md                     # Comprehensive documentation & 20 interview Q&As
└── requirements.txt              # Project dependencies
```

---

## ⚡ 3. Quickstart & Usage

### 3.1 Installation
```powershell
pip install -r requirements.txt
```

### 3.2 Running an Interactive Query
```powershell
$env:PYTHONPATH="Day 118"
python "Day 118/app/main.py" -q "What does an HTTP 404 error indicate?"
```
*Output:*
```text
Loading knowledge base from Day 118/data/raw...
Indexed 30 documents into 93 chunks.

Processing query: 'What does an HTTP 404 error indicate?' (Method: hybrid_rrf)

============================================================
ANSWER:
Based on the provided documentation: The HTTP 404 Not Found client error status response code indicates that the web server cannot locate the requested resource at the specified Uniform Resource Identifier (URI). Unlike HTTP 401 Unauthorized or 403 Forbidden, a 404 indicates the endpoint or file path does not exist in the routing table. [Source: HTTP 404 Error Status Code and Web Server Routing - Chunk 0]

**Sources & Citations:**
1. [Source: HTTP 404 Error Status Code and Web Server Routing - Chunk 0] (Score: 0.6054): "The HTTP 404 Not Found client error status response code indicates..."
============================================================
Supported: True
Compression Ratio: 30.3%
Latency: 2.1 ms
```

### 3.3 Executing the Complete Benchmark Suite (7 Experiments + 15 Charts)
```powershell
$env:PYTHONIOENCODING="utf-8"
$env:MPLCONFIGDIR="$env:TEMP"
$env:PYTHONPATH="Day 118"
python "Day 118/experiments/run_experiments.py"
```

### 3.4 Running All 101 Unit Tests
```powershell
$env:PYTHONPATH="Day 118"
pytest "Day 118/tests" -v
```

---

## 🧪 4. The 7 Coding Challenges (Implemented in `scratch/`)

| Challenge Script | Target Technique | Verification Criteria |
| :--- | :--- | :--- |
| [`scratch_rrf.py`](file:///s:/Programming/Python200days/Day%20118/scratch/scratch_rrf.py) | **Reciprocal Rank Fusion** | Aggregates 3 heterogeneous rank lists without score calibration using $k_0 = 60$. |
| [`scratch_hybrid.py`](file:///s:/Programming/Python200days/Day%20118/scratch/scratch_hybrid.py) | **Weighted Score Hybrid** | Applies min-max normalization across disparate scales ($[0, \infty)$ vs $[-1, 1]$). |
| [`scratch_expansion.py`](file:///s:/Programming/Python200days/Day%20118/scratch/scratch_expansion.py) | **Query Optimization** | Strips conversational prefixes and expands domain synonyms (*"overfitting"*). |
| [`scratch_mmr.py`](file:///s:/Programming/Python200days/Day%20118/scratch/scratch_mmr.py) | **Maximal Marginal Relevance** | Suppresses identical duplicate chunks while preserving diverse evidence at $\lambda = 0.5$. |
| [`scratch_metadata.py`](file:///s:/Programming/Python200days/Day%20118/scratch/scratch_metadata.py) | **Metadata Pre-Filtering** | Implements composable predicates with token limits, category matching, and tags. |
| [`scratch_compressor.py`](file:///s:/Programming/Python200days/Day%20118/scratch/scratch_compressor.py) | **Context Compression** | Filters out boilerplate lines and achieves $>60\%$ prompt character reduction. |
| [`scratch_benchmark.py`](file:///s:/Programming/Python200days/Day%20118/scratch/scratch_benchmark.py) | **IR Micro-Benchmark** | Evaluates Precision@3, Recall@3, MRR, and nDCG@3 on graded relevance labels. |

---

## 📊 5. Empirical Benchmark Summary Table

Derived directly from 60 benchmark evaluation queries across 93 indexed chunks:

| Model | P@5 | R@5 | Hit Rate@5 | MRR | nDCG@5 | Median Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Sparse)** | 0.1964 | 0.9818 | 98.2% | 0.8885 | 0.9090 | **0.13 ms** |
| **Dense (Cosine SVD)** | **0.2000** | **1.0000** | **100.0%** | **0.9179** | **0.9348** | 0.93 ms |
| **Hybrid (Weighted Min-Max)** | **0.2000** | **1.0000** | **100.0%** | 0.9121 | 0.9338 | 1.12 ms |
| **Hybrid (RRF, $k_0=60$)** | 0.1964 | 0.9818 | 98.2% | 0.8793 | 0.9051 | 1.19 ms |
| **Hybrid + Rerank** | 0.1927 | 0.9636 | 96.4% | 0.8699 | 0.8930 | 2.23 ms |
| **Advanced RAG (Full)** | 0.1927 | 0.9636 | 96.4% | 0.8881 | 0.9114 | 30.55 ms |

---

## 🎓 6. 20 Technical Interview Questions & Answers on Advanced RAG

### Q1: What is the fundamental difference between sparse and dense retrieval?
**Answer:**  
Sparse retrieval (e.g., BM25, TF-IDF) represents queries and documents as high-dimensional, sparse vectors where each dimension corresponds to an exact vocabulary term. Term weights are determined by frequency and inverse collection rarity. Dense retrieval maps text into low-dimensional continuous geometric spaces (64 to 1536 dimensions) using deep neural representations. Sparse retrieval excels at exact keywords, IDs, and error codes but suffers from vocabulary mismatch. Dense retrieval captures semantic synonyms and paraphrases but can struggle with precise alphanumeric codes or rare domain entities.

---

### Q2: How does Okapi BM25 improve upon classical TF-IDF?
**Answer:**  
Classical TF-IDF uses linear or logarithmic term frequency weights that grow without bound as a term repeats, allowing a spammy or overly verbose document to accumulate an artificially inflated score. Okapi BM25 introduces **non-linear term frequency saturation** parameterized by $k_1$:
$$\frac{f(q_i, D) (k_1 + 1)}{f(q_i, D) + k_1 \cdot \text{len\_norm}}$$
As $f(q_i, D)$ increases, the marginal score increase approaches an asymptote $(k_1 + 1)$. Furthermore, BM25 normalizes by document length ($b$ parameter), penalizing long documents that contain terms merely due to length rather than focused topical relevance.

---

### Q3: What is the role of hyperparameters $k_1$ and $b$ in BM25?
**Answer:**  
- **$k_1$ (typically 1.2 to 2.0, default 1.5):** Controls the term frequency saturation curve. Higher values allow repeated terms to contribute more marginal score before reaching the asymptotic ceiling; lower values make the score saturate rapidly.
- **$b$ (typically 0.75, range $[0.0, 1.0]$):** Controls document length normalization. At $b = 1.0$, long documents are heavily penalized for verbosity; at $b = 0.0$, document length normalization is completely disabled, treating documents of all lengths equally.

---

### Q4: What is Reciprocal Rank Fusion (RRF), and why is it preferred over weighted score combination?
**Answer:**  
Linear score combination requires min-max normalizing scores: $\text{Score} = \alpha S_{\text{dense}} + (1 - \alpha) S_{\text{bm25}}$. However, BM25 scores are unbounded and skewed, while dense cosine scores are bounded and clustered in a narrow band. Extreme outlier scores in one retriever can distort the fused ranking.  
Reciprocal Rank Fusion (RRF) discards raw scores entirely and operates solely on ordinal rank positions:
$$\operatorname{RRF}(d) = \sum_{r \in R} \frac{1}{k_0 + \operatorname{rank}_r(d)}$$
RRF is unsupervised, parameter-free, scale-invariant, robust to score distribution shifts, and consistently outperforms score combination across heterogeneous retrieval engines.

---

### Q5: What is the significance of the smoothing constant $k_0 = 60$ in RRF?
**Answer:**  
The constant $k_0 = 60$ was empirically derived by Cormack, Clarke, and Büttcher (SIGIR 2009). Its purpose is to prevent top ranks (rank 1 vs rank 2) from exerting disproportionate dominance over the aggregate score. With $k_0 = 0$, rank 1 gets weight $1.0$, while rank 2 gets $0.5$ (a massive 50% drop). With $k_0 = 60$, rank 1 gets $\frac{1}{61} \approx 0.01639$ and rank 2 gets $\frac{1}{62} \approx 0.01613$ (a smooth 1.6% delta), ensuring that a document appearing near the top across multiple rankers comfortably beats a document that ranked first in only one list.

---

### Q6: How do Bi-Encoders and Cross-Encoders differ architecturally?
**Answer:**  
- **Bi-Encoder:** The query $q$ and document $d$ are passed through separate transformer encoders (or the same encoder independently) to produce dense vectors $u = f(q)$ and $v = g(d)$. Relevance is scored via cosine similarity or dot product $u \cdot v$. Document embeddings can be precomputed and indexed in vector databases (e.g., HNSW).
- **Cross-Encoder:** The query and document are concatenated as a single input sequence `[CLS] query [SEP] document [SEP]` and processed simultaneously through all transformer layers. Every query token attends directly to every document token via full bidirectional cross-attention. Cross-encoders achieve significantly higher ranking precision but require $O(N)$ transformer forward passes at runtime.

---

### Q7: Why are Cross-Encoders used only as second-stage rerankers rather than first-stage retrievers?
**Answer:**  
In a corpus with 1,000,000 documents, a bi-encoder computes 1 query embedding and executes an approximate nearest neighbor search in $<5\text{ ms}$ because document vectors are precomputed. A cross-encoder would need to concatenate the query with all 1,000,000 documents and perform 1,000,000 full forward passes through deep transformer layers, taking minutes per query. Therefore, production systems use a fast first-stage retriever (BM25 / Dense) to retrieve the top 20-50 candidates, which are then reranked by the cross-encoder in $<20\text{ ms}$.

---

### Q8: What is Maximal Marginal Relevance (MMR), and what problem does it solve in RAG?
**Answer:**  
In standard top-K retrieval, the highest-scoring chunks are often nearly identical paragraphs from consecutive sections of the same document. Passing near-duplicates to an LLM wastes prompt token budget and crowds out diverse, complementary facts. MMR resolves this by iteratively selecting chunks that maximize query relevance while penalizing similarity to already selected chunks:
$$\operatorname{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \operatorname{Sim}_1(d_i, Q) - (1 - \lambda) \max_{d_j \in S} \operatorname{Sim}_2(d_i, d_j) \right]$$

---

### Q9: How should the $\lambda$ parameter in MMR be tuned in production?
**Answer:**  
- $\lambda = 1.0$: Pure relevance (standard greedy top-K, potential high redundancy).
- $\lambda = 0.0$: Pure diversity (novelty maximized, relevance ignored).
- Empirical benchmarks show that $\lambda \in [0.6, 0.7]$ represents the sweet spot, preserving $>97\%$ of top-K relevance while reducing inter-chunk duplicate text by $30\text{--}40\%$.

---

### Q10: What is the difference between metadata pre-filtering and post-filtering?
**Answer:**  
- **Pre-filtering:** Filters the document corpus *before* vector index traversal or BM25 scoring. Only chunks matching metadata criteria (e.g., `user_id = 123`, `category = 'dbms'`) are evaluated. Guarantees that all returned top-K results satisfy the criteria and reduces search latency.
- **Post-filtering:** Retrieves top $N$ candidates across the entire index first, then filters out candidates that do not match the metadata criteria. If the matching documents are ranked lower than $N$, post-filtering can return fewer than $K$ results or even an empty set.

---

### Q11: When can metadata pre-filtering degrade retrieval performance?
**Answer:**  
In vector databases utilizing graph-based indexes (such as HNSW), naive pre-filtering can disconnect the proximity graph if many nodes are filtered out, trapping the greedy search in local minima. Advanced vector databases solve this using **filtered single-stage vector traversal** (accommodating filter bitmaps during HNSW beam search) or **payload inverted indexes**.

---

### Q12: What is Context Compression in RAG, and why is it beneficial beyond token cost savings?
**Answer:**  
Context compression extracts only the relevant sentences or spans from retrieved chunks, discarding boilerplate intros, headers, and tangential prose. Beyond saving LLM token API costs, it directly mitigates the **"Lost in the Middle"** phenomenon (Liu et al., 2023), where LLM attention is sharpest at the beginning and end of the prompt context and severely degraded in the middle. Dense, compressed context ensures key facts are not obscured.

---

### Q13: How does Query Rewriting differ from Query Expansion?
**Answer:**  
- **Query Rewriting:** Transforms conversational or ambiguous user questions into clean, keyword-dense search queries by stripping conversational boilerplate (*"Could you please tell me..."*), correcting spelling, and disambiguating references (e.g., replacing pronouns using chat history).
- **Query Expansion:** Enriches the query by adding domain-specific synonyms, related entities, or conceptual terms to bridge the vocabulary mismatch gap (*"overfitting"* $\to$ *"overfitting generalization validation loss regularization"*).

---

### Q14: How does Multi-Query Retrieval operate?
**Answer:**  
A complex user prompt is decomposed into 3-4 diverse subqueries representing different perspectives (e.g., theoretical definition, implementation steps, troubleshooting). Each subquery is executed independently against the retriever, producing separate candidate lists. The candidate lists are then unified using Reciprocal Rank Fusion (RRF) to produce a single, diverse candidate pool.

---

### Q15: What is nDCG@K, and why is it preferred over binary Precision@K or Recall@K?
**Answer:**  
Precision@K and Recall@K treat relevance as a binary attribute (relevant vs irrelevant) and ignore the position of relevant items within the top-K. A system that places the best document at Rank 1 gets the exact same Precision@5 as a system that places it at Rank 5.  
Normalized Discounted Cumulative Gain (nDCG@K) incorporates **graded relevance** ($rel \in \{0, 1, 2, 3\}$) and applies logarithmic position discounting $\frac{1}{\log_2(i + 1)}$. Placing a highly relevant document at Rank 1 yields significantly higher nDCG than placing it at Rank 5.

---

### Q16: How do you mathematically prevent Recall@K from exceeding 1.0 when multiple chunks belong to the same document?
**Answer:**  
If a query targets document $D$ and the system retrieves 3 separate chunks from $D$ in top-5, counting raw chunk matches would yield $3 / 1 = 300\%$ recall. In document-level evaluation, chunks must be mapped to unique document IDs:
$$\operatorname{Recall@K} = \frac{|\{d \in \operatorname{top\_k\_doc\_ids}\} \cap \operatorname{RelevantDocs}|}{|\operatorname{RelevantDocs}|}$$
ensuring Recall is strictly bounded in $[0.0, 1.0]$.

---

### Q17: How does a RAG pipeline reliably detect and reject unanswerable queries?
**Answer:**  
The pipeline computes evidence confidence thresholds:
1. Candidate retrieval scores (e.g., maximum dense cosine similarity or cross-encoder score) must exceed a minimum threshold (e.g., 0.05).
2. Lexical keyword overlap between question tokens and retrieved chunk text must exceed a minimum ratio (e.g., 25%).
3. System prompt instructions explicitly enforce: *"If the answer cannot be determined from the provided context, state: 'I don't have enough information in the provided documents to answer this question.'"*

---

### Q18: What is the difference between extractive and abstractive context compression?
**Answer:**  
- **Extractive Compression:** Scores individual sentences within retrieved chunks and retains only high-scoring sentences verifiably identical to source text. Fast ($<1\text{ ms}$), deterministic, and carries zero hallucination risk.
- **Abstractive Compression:** Uses a lightweight LLM (e.g., T5 or a small SLM) to summarize retrieved chunks into concise paragraphs. Flexible and synthesizes disparate points, but adds latency ($100\text{--}300\text{ ms}$) and introduces hallucination risk.

---

### Q19: In what practical scenarios does BM25 outperform state-of-the-art dense semantic embedding models?
**Answer:**  
1. **Rare Technical Identifiers:** Error codes (`SQLSTATE 23505`, `HTTP 404`), UUIDs, product SKUs, git commit hashes.
2. **Domain-Specific Acronyms:** Highly specialized industry abbreviations not well-represented in general pre-training corpora.
3. **Exact Variable / Method Names:** Code search for functions like `batch_cosine_similarity`.
4. **Ultra-Low Latency Constraints:** BM25 executes in $<0.2\text{ ms}$ on CPU without requiring GPU acceleration or floating-point matrix multiplication.

---

### Q20: What is the optimal sequence of operations in a production Advanced RAG pipeline?
**Answer:**  
1. **Query Processing:** Rewrite conversational prompt + expand domain synonyms.
2. **Stage 1 Hybrid Retrieval:** Parallel BM25 + Dense vector search with metadata pre-filtering (fetch top 20-30 candidates).
3. **Rank Fusion:** Merge candidate lists using Reciprocal Rank Fusion ($k_0 = 60$).
4. **Stage 2 Reranking:** Re-score top 20 candidates using a Cross-Encoder.
5. **Diversity Pruning:** Apply MMR ($\lambda = 0.7$) to select the top 5 diverse chunks.
6. **Context Compression:** Extractively compress the 5 chunks to eliminate filler sentences.
7. **Grounded Generation:** Prompt LLM with strict anti-hallucination instructions and bracketed source attribution.
