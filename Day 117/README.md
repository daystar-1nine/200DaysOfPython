# Day 117: Retrieval-Augmented Generation (RAG) Architecture & Document Q&A Engine

**Python 200 Days Curriculum — Day 117 / 200 (58.5% Complete — 83 Days Remaining)**  
**Author:** Suraj Sawant (`daystar-1nine`)

---

## 📌 Overview

Day 117 implements a complete, dependency-free **Mini RAG (Retrieval-Augmented Generation) Document Q&A Engine** from first principles in Python.

While Large Language Models excel at reasoning, they suffer from **hallucinations**, **knowledge cutoffs**, and **zero access to private data**. RAG resolves this by decoupling non-parametric factual retrieval from parametric linguistic synthesis:

$$\text{User Query} \longrightarrow \text{Retrieve Relevant Chunks} \longrightarrow \text{Construct Grounded Prompt} \longrightarrow \text{Generate Answer with Citations}$$

```text
                  ┌──────────────────────────────────────────────┐
                  │   Knowledge Base (30 Multi-Domain Docs)      │
                  │   Python, Data Science, ML, DL, CV, NLP, DBMS│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                            [ Document Preprocessor ]
                            (Unicode NFC, Strip Noise)
                                         │
                                         ▼
                            [ Sentence-Aware Chunker ]
                            (Size=400 chars, Overlap=50)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          [ Lexical Inverter ]                      [ Dense TF-IDF + SVD ]
         (BM25 Term Frequencies)                    (64-dim L2 Unit Vectors)
                   │                                           │
                   ▼                                           ▼
          [ Lexical BM25 Store ]                    [ Vector Index Store ]
                   │                                           │
User Query ────────┴───────────────────────────────────────────┘
     │
     ├──► BM25 Query Search ────────────┐
     │                                  ▼
     └──► Vector Cosine Search ──► [ Hybrid Engine ] (RRF / MinMax Fusion)
                                        │
                                        ▼
                             Top-K Relevant Chunks
                                        │
                                        ▼
                           [ Anti-Hallucination Audit ]
                          (Overlap >= 20% OR Score >= 0.60)
                                        │
                         ┌──────────────┴──────────────┐
                    Passed                             Failed (< Threshold)
                         ▼                                     ▼
             [ Context Assembly ]                  [ Canonical Abstention ]
            (Bounded Char Budget)                "I don't have enough info..."
                         │                                     │
                         ▼                                     │
             [ Grounded Generator ]                            │
            (Extractive Synthesis)                             │
                         │                                     │
                         ▼                                     │
             [ Verifiable Citations ] ◄────────────────────────┘
          [Source: Title - Chunk N]
```

---

## 📁 Repository Directory Structure

```text
Day 117/
├── app/
│   ├── citations/          # Citation tag formatting & source verification
│   ├── chunking/           # Character, word, and sentence-level text chunkers
│   ├── embeddings/         # Dense TF-IDF + SVD, sparse TF-IDF, cosine similarity
│   ├── evaluation/         # Precision@K, Recall@K, MRR, Faithfulness, Hallucination
│   ├── generation/         # Prompt builder, context construction, answer generator
│   ├── index/              # In-memory vector store & disk serialization (.npz / .json)
│   ├── ingestion/          # Document loader, metadata extraction, text cleaner
│   ├── reporting/          # 15 publication-grade visualization generators
│   ├── retrieval/          # BM25 Keyword, Semantic, and Hybrid retrievers
│   ├── config.py           # System-wide configuration dataclass
│   ├── main.py             # CLI entrypoint (ingest, query, evaluate)
│   └── pipeline.py         # Unified end-to-end RAG pipeline
│
├── data/
│   ├── raw/                # 30 Markdown documents across 7 technical categories
│   ├── evaluation/         # eval_questions.json (60 ground-truth benchmark queries)
│   └── generate_knowledge_base.py # Knowledge base & ground truth generator
│
├── experiments/
│   ├── chunk_size.csv      # Experiment 1: 200, 400, 600, 800 char sensitivity
│   ├── top_k.csv           # Experiment 2: K = 1, 3, 5, 10 retrieval trade-off
│   ├── retrieval_methods.csv # Experiment 3: Keyword vs Semantic vs Hybrid
│   ├── hybrid_search.csv   # Experiment 4: Query type breakdown
│   └── run_experiments.py  # Automated experiment runner
│
├── index/                  # Persisted vector array (vectors.npz) and chunk metadata
│
├── outputs/
│   ├── charts/             # 15 publication-quality analytical PNG figures
│   ├── error_analysis.csv  # Categorical taxonomy of benchmark errors
│   ├── metrics.csv         # Aggregated benchmark evaluation metrics
│   ├── predictions.csv     # Model answers and source citations for all 60 queries
│   └── retrieval_results.csv # Query-level retrieval scores and ranking metrics
│
├── scratch/                # 7 Coding Challenges from scratch
│   ├── scratch_chunking.py # Challenge 1: Sliding window text chunker
│   ├── scratch_cosine.py   # Challenge 2: Pure math & numpy cosine similarity
│   ├── scratch_retrieval.py# Challenges 3-6: Top-K, Precision@K, Recall@K, MRR
│   └── scratch_rag.py      # Challenge 7: Standalone Mini Semantic Search Engine
│
├── tests/                  # 139 passing unit tests
├── DAY_117_REPORT.md       # Comprehensive technical report
├── README.md               # Complete documentation & 30 interview Q&As
└── requirements.txt        # Runtime dependencies
```

---

## ⚡ Quickstart Guide

### 1. Installation
```powershell
pip install -r "Day 117/requirements.txt"
$env:PYTHONPATH="Day 117"
```

### 2. Ingest & Index Knowledge Base
Ingests 30 technical markdown files into 110 indexed chunks:
```powershell
python -m app.main ingest
```

### 3. Query the Engine
```powershell
python -m app.main query --question "What is the Global Interpreter Lock (GIL) in CPython?" --method hybrid --top-k 3
```

Sample output:
```text
======================================================================
  [Mini RAG Query] Method: HYBRID | Top-K: 3
======================================================================
Question: What is the Global Interpreter Lock (GIL) in CPython?

----------------------------------------------------------------------
Answer:
The CPython implementation relies on the Global Interpreter Lock (GIL), a mutual-exclusion lock preventing multiple native operating system threads from executing Python bytecode concurrently. While the GIL protects CPython memory management and reference counting from race conditions, it limits CPU-bound multi-threading to a single physical core.

Sources:
- [Source: Global Interpreter Lock (GIL) and Concurrency in Python - Chunk 0] (Relevance: 0.88)
----------------------------------------------------------------------
Supported Grounding: True
Latency: 0.95 ms
```

### 4. Run Benchmark Evaluation
Evaluates 60 ground-truth questions against Precision@K, Recall@K, MRR, and Faithfulness:
```powershell
python -m app.main evaluate --method hybrid --top-k 3
```

### 5. Run Full Experiments Suite
Executes all 4 experiments and generates 15 publication charts:
```powershell
python "Day 117/experiments/run_experiments.py"
```

### 6. Run Test Suite
```powershell
pytest "Day 117/tests" -v
# 139 passed in 0.31s
```

---

## 💡 7 Coding Challenges & First-Principles Implementations

All 7 coding challenges from Day 117 are implemented and verified in `Day 117/scratch/`:

### Challenge 1: Sliding Window Chunking with Overlap
*File: [`scratch_chunking.py`](file:///s:/Programming/Python200days/Day%20117/scratch/scratch_chunking.py)*
```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")
    stride = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), stride)]
```

### Challenge 2: Pure Cosine Similarity (Zero External Libraries)
*File: [`scratch_cosine.py`](file:///s:/Programming/Python200days/Day%20117/scratch/scratch_cosine.py)*
```python
import math

def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
```

### Challenges 3, 4, 5, 6: Retrieval & Evaluation Metrics
*File: [`scratch_retrieval.py`](file:///s:/Programming/Python200days/Day%20117/scratch/scratch_retrieval.py)*
```python
import numpy as np

# Challenge 3: Top-K Vector Search
def top_k_similar(query_vec, doc_vecs, k=3):
    sims = np.dot(doc_vecs, query_vec) / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec))
    top_indices = np.argsort(sims)[::-1][:k]
    return [(idx, float(sims[idx])) for idx in top_indices]

# Challenge 4: Precision@K
def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / k

# Challenge 5: Recall@K
def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant) if relevant else 0.0

# Challenge 6: Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(all_retrieved, all_relevant):
    rr = [1.0 / (ret.index(rel[0]) + 1) if rel[0] in ret else 0.0 for ret, rel in zip(all_retrieved, all_relevant)]
    return sum(rr) / len(rr)
```

### Challenge 7: Standalone Mini Semantic Search Engine (No LLM)
*File: [`scratch_rag.py`](file:///s:/Programming/Python200days/Day%20117/scratch/scratch_rag.py)*  
Implements self-contained document ingestion, TF-IDF vocabulary building, L2 unit vectorization, in-memory cosine index, and top-K search with formatted snippets.

---

## 📊 Summary of Empirical Results

| Metric | Keyword (BM25) | Semantic (Dense) | Hybrid (RRF) |
|:---|:---:|:---:|:---:|
| **Precision@1** | 0.7667 | 0.7667 | **0.7667** |
| **Precision@3** | 0.4500 | **0.4667** | 0.4611 |
| **Recall@3** | 0.6750 | **0.7000** | 0.6917 |
| **Mean Reciprocal Rank (MRR)** | 0.7750 | **0.7806** | 0.7750 |
| **Hit Rate@5** | 0.7833 | **0.8000** | 0.7833 |
| **Faithfulness** | 0.8786 | 0.8842 | **0.8851** |
| **Context Relevance** | **0.6614** | 0.6300 | 0.6383 |
| **Citation Coverage** | 100% | 100% | **100%** |
| **Query Latency** | **0.14 ms** | 0.84 ms | 0.95 ms |

---

## 🎓 30 Technical Interview Questions & Answers

### 1. Fundamentals of RAG & Hallucination Mitigation
**Q1: What is the fundamental difference between parametric and non-parametric memory in LLMs?**  
*Answer:* Parametric memory refers to the knowledge baked into the billions of neural weights of a model during pretraining; it is static, computationally expensive to update, and opaque. Non-parametric memory refers to external data stores (vector databases, relational tables, documents) that can be dynamically searched, updated in real time with zero training cost, and explicitly cited.

**Q2: What is Retrieval-Augmented Generation (RAG) and why is it preferred over fine-tuning for factual knowledge updates?**  
*Answer:* RAG dynamically retrieves relevant factual documents at inference time and injects them into the model's context window. It is preferred over fine-tuning because: (1) fine-tuning is prone to catastrophic forgetting and hallucinations; (2) updating non-parametric vector stores takes seconds rather than hours of GPU compute; (3) RAG provides verifiable source citations; (4) enterprise access controls and permissions can be enforced at the retrieval layer.

**Q3: What causes LLM hallucinations and how does RAG mitigate them?**  
*Answer:* Hallucinations occur because autoregressive models optimize for next-token probability rather than factual truth, generating plausible-sounding continuations when uncertain. RAG mitigates hallucinations by grounding generation on retrieved source context, instructing the model to rely solely on the provided context and refuse to answer if information is absent.

**Q4: What is the "Lost in the Middle" phenomenon in RAG context windows?**  
*Answer:* Research (Liu et al., 2023) demonstrated that LLMs attend effectively to tokens located at the very beginning and very end of their context window, but performance sharply degrades when crucial factual evidence is placed in the middle. Context builders should order the most relevant chunks at the beginning and end.

**Q5: What are the primary failure modes of a RAG pipeline?**  
*Answer:* (1) **Retrieval failure:** Missing the relevant document entirely (low recall). (2) **Irrelevant retrieval:** Fetching noisy, distracting chunks (low precision). (3) **Context truncation:** Context window overflow truncating critical evidence. (4) **Ungrounded generation:** Generator ignoring retrieved context and hallucinating. (5) **Refusal failure:** Generator guessing instead of abstaining on unsupported queries.

---

### 2. Chunking & Ingestion Strategies
**Q6: Why is document chunking necessary in RAG instead of feeding entire documents into the LLM?**  
*Answer:* Even models with 1M token context windows suffer from computational quadratic attention scaling, latency inflation, and "Lost in the Middle" degradation. Small, focused chunks maximize semantic vector specificity and minimize distracting irrelevant text.

**Q7: How does chunk size affect retrieval precision and recall?**  
*Answer:* Small chunks (e.g. 100–200 chars) yield high retrieval precision because embeddings are focused on a single proposition, but suffer from low recall due to context fragmentation. Large chunks (e.g. 1000+ chars) capture complete context (higher recall) but dilute semantic vector representations (lower precision).

**Q8: What is chunk overlap and why is it critical?**  
*Answer:* Chunk overlap is the sliding window boundary where consecutive chunks share trailing/leading tokens. It prevents splitting critical phrases, sentences, or concepts across chunk boundaries, ensuring that ideas spanning the boundary remain intact in at least one chunk.

**Q9: What is semantic chunking compared to fixed-size chunking?**  
*Answer:* Fixed-size chunking slices text strictly by character or token count. Semantic chunking dynamically splits text based on semantic shifts, such as markdown header hierarchies, paragraph boundaries, or cosine distance spikes between consecutive sentence embeddings.

**Q10: What metadata should be stored alongside document chunks?**  
*Answer:* `document_id`, `chunk_id`, `chunk_index`, `title`, `category`, `source_url`/`file_path`, `start_char`/`end_char` spans, creation timestamp, and access control tags (user permissions).

---

### 3. Embeddings & Vector Space Geometry
**Q11: Why is cosine similarity universally preferred over Euclidean distance for text embeddings?**  
*Answer:* Cosine similarity measures the angle between vectors, reflecting directional semantic alignment independent of vector magnitude (text length). When vectors are normalized to unit length (\(\|\mathbf{v}\|_2 = 1\)), cosine similarity is linearly equivalent to squared Euclidean distance: $\|\mathbf{a} - \mathbf{b}\|^2 = 2 - 2 \cos(\mathbf{a}, \mathbf{b})$.

**Q12: What is the curse of dimensionality and how does it affect vector similarity search?**  
*Answer:* In high-dimensional spaces (e.g. 768 or 1536 dims), the ratio between the distance to the nearest neighbor and the distance to the farthest neighbor approaches 1.0, making all vectors appear almost equidistant. High-quality pretrained embeddings counteract this by confining semantic representations to lower-dimensional non-linear manifolds.

**Q13: What is the difference between dense embeddings and sparse embeddings?**  
*Answer:* Dense embeddings (e.g. OpenAI `text-embedding-3`, BGE) are low-dimensional (256–1536 floats) continuous vectors where every dimension is non-zero, capturing latent semantic concepts and synonyms. Sparse embeddings (e.g. BM25, SPLADE) have thousands of dimensions (vocabulary size) where only specific matching word tokens have non-zero weights, capturing exact keywords.

**Q14: What is out-of-vocabulary (OOV) handling in dense embedders vs lexical search?**  
*Answer:* Lexical BM25 search cannot score unseen terms (score = 0). Dense neural embedders handle OOV words via subword tokenization (BPE/WordPiece), breaking unseen words into known morphological subwords to produce continuous vector approximations.

**Q15: How does MRL (Matryoshka Representation Learning) work in modern embedding models?**  
*Answer:* Matryoshka embeddings are trained so that the first $d$ dimensions (e.g. 64 or 256) retain most of the semantic fidelity of the full vector (e.g. 1536). This allows truncating embeddings to save 80–90% storage and RAM while sacrificing less than 2% retrieval accuracy.

---

### 4. Vector Databases & Indexing Algorithms
**Q16: How does Approximate Nearest Neighbor (ANN) search differ from Exact Nearest Neighbor ($k$-NN) search?**  
*Answer:* Exact $k$-NN computes distances against all $N$ vectors ($O(N \cdot D)$ time complexity), which is prohibitively slow for millions of vectors. ANN trades a tiny fraction of recall (e.g. 98% instead of 100%) for logarithmic $O(\log N)$ or sub-linear search time using indexed spatial data structures.

**Q17: How does HNSW (Hierarchical Navigable Small World) graph indexing work?**  
*Answer:* HNSW constructs a multi-layer graph hierarchy analogous to a skip list. Upper layers contain sparse long-range connections for fast coarse geographic traversal, while bottom layers contain dense clusters of nearest neighbors for fine-grained localized routing.

**Q18: What is Inverted File Index with Product Quantization (IVF-PQ)?**  
*Answer:* IVF partitions the vector space into Voronoi cells using $k$-means clustering, searching only the nearest cluster centroids. Product Quantization (PQ) compresses vectors into small quantized byte codes, reducing memory consumption by 80–95% and accelerating distance computation via lookup tables.

**Q19: What are the trade-offs between dedicated vector databases (Pinecone, Qdrant, Milvus) and relational/search extensions (pgvector, Elasticsearch)?**  
*Answer:* Dedicated vector DBs offer state-of-the-art ANN performance, native quantization, and massive horizontal scaling. Database extensions (pgvector) eliminate data sync pipelines, support ACID transactional consistency, and enable unified relational filtering, but may scale less efficiently beyond tens of millions of high-dimensional vectors.

**Q20: How does metadata pre-filtering differ from post-filtering in vector databases?**  
*Answer:* Post-filtering performs vector search first and discards chunks that fail metadata predicates; if the top-$K$ chunks are discarded, fewer than $K$ results are returned. Pre-filtering filters candidates by metadata before vector scoring, guaranteeing that all returned chunks meet the criteria.

---

### 5. Hybrid Retrieval & Re-ranking
**Q21: Why does Hybrid Search outperform both pure keyword and pure semantic search?**  
*Answer:* Keyword search (BM25) excels at exact alphanumeric matches (product codes, error logs, acronyms) but fails on synonyms. Semantic search excels at conceptual intent and paraphrasing but misses rare exact keywords. Hybrid search combines the strengths of both.

**Q22: How does Reciprocal Rank Fusion (RRF) combine ranking lists?**  
*Answer:* RRF merges multiple ranking lists without requiring score normalization by summing reciprocal ranks:
$$RRF\_Score(d) = \sum_{m \in \text{Models}} \frac{1}{k + \text{Rank}_m(d)}$$
where $k \approx 60$ dampens the influence of top ranks, ensuring robust consensus.

**Q23: What is Cross-Encoder Re-ranking and why is it used as a second stage in RAG?**  
*Answer:* Bi-encoders embed query and document independently to allow fast vector lookups, but miss fine-grained token-level cross-attention. A Cross-Encoder re-ranker passes `(query, document)` together through a Transformer model, allowing full all-to-all attention to re-score the top 50 candidates with high accuracy.

**Q24: What is Query Rewriting / Expansion in advanced RAG?**  
*Answer:* User queries are frequently ambiguous, short, or poorly phrased. A query rewriter uses an LLM to generate multiple paraphrased variations, decompose complex queries into sub-queries, or produce hypothetical document embeddings (HyDE).

**Q25: What is Hypothetical Document Embeddings (HyDE)?**  
*Answer:* HyDE instructs an LLM to generate a hypothetical answer to the user query. Even if factually inaccurate, this hypothetical document possesses the semantic tone, structure, and domain vocabulary of the target answer, making its dense embedding closer to real relevant documents in vector space than the raw query.

---

### 6. Grounding, Evaluation & Guardrails
**Q26: What are the three core RAG evaluation metrics in the Ragas framework?**  
*Answer:*
1. **Faithfulness:** Are all statements in the generated answer supported by the retrieved context?
2. **Answer Relevance:** Does the answer address the question asked?
3. **Context Relevance:** Are the retrieved context chunks focused on the question without noise?

**Q27: How are Precision@K, Recall@K, and MRR calculated for retrieval benchmarking?**  
*Answer:*
- $\text{Precision@K} = \frac{|\text{Retrieved[:K]} \cap \text{Relevant}|}{K}$
- $\text{Recall@K} = \frac{|\text{Retrieved[:K]} \cap \text{Relevant}|}{|\text{Relevant}|}$
- $\text{MRR} = \frac{1}{Q} \sum_{i=1}^Q \frac{1}{\text{Rank of First Relevant Doc}}$

**Q28: How do you design an anti-hallucination guardrail for out-of-domain queries?**  
*Answer:* Implement a thresholding gate before generation: (1) if the highest retrieval score is below a calibrated confidence floor (e.g. $< 0.08$), or (2) if key query terms have zero overlap with the retrieved text, bypass generation and emit a canonical abstention: *"I don't have enough information in the provided documents to answer this question."*

**Q29: How do you implement verifiable citation attribution in RAG?**  
*Answer:* Tag each retrieved chunk with unique metadata (`[Source: Document Title - Chunk N]`), instruct the generator to reference these tags when stating facts, and verify post-generation that all cited tags exist in the provided context and cover the claims made.

**Q30: What is the difference between RAG and Long-Context LLMs (e.g. Gemini 1.5 Pro with 2M tokens)?**  
*Answer:* Long-context LLMs can ingest an entire book or codebase in a single prompt, but cost significantly more in inference latency, dollar cost per prompt, and token processing. RAG remains essential for: (1) sub-second latency, (2) cost efficiency (retrieving 3 chunks vs sending 2M tokens), (3) real-time data updates without re-sending massive contexts, and (4) strict enterprise access control.
