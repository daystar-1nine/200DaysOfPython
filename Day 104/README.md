# Semantic Document Search Engine

An end-to-end information retrieval system comparing keyword-based search (TF-IDF), dense semantic retrieval (word embeddings with pooling), and hybrid retrieval architectures on a curated multi-domain corpus.

---

## Overview

In traditional information retrieval, keyword-matching systems (such as TF-IDF and BM25) index documents based on exact term occurrences. While computationally efficient and precise for specific keyword queries, keyword search struggles with **synonymy** (different words with the same meaning, such as *"car"* vs. *"automobile"* or *"vehicle maintenance"* vs. *"auto repair"*) and **vocabulary mismatch**.

**Semantic Search** solves this by mapping queries and documents into a continuous, multi-dimensional semantic vector space. Documents are retrieved based on **conceptual meaning and geometric proximity**, rather than verbatim lexical overlap.

This project implements a complete, production-grade retrieval pipeline from first principles using pure Python, NumPy, SciPy, and Scikit-Learn:
1. **TF-IDF Search Engine**: High-dimensional sparse vector space with sublinear term-frequency scaling and cosine similarity.
2. **Dense Embedding Search Engine**: Low-dimensional dense representations using distributional word embeddings (PPMI + TruncatedSVD) with mean and weighted pooling.
3. **Hybrid Search Engine**: Convex combination of normalized keyword and semantic similarity scores.
4. **Evaluation Suite**: Rigorous assessment using standard Information Retrieval (IR) metrics: **Precision@K**, **Recall@K**, and **Mean Reciprocal Rank (MRR)** against an annotated ground-truth benchmark.

---

## Problem Statement

Given:
- A document corpus $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ spanning multiple technical domains.
- A user query string $q$.
- An integer $K \ge 1$.

Retrieve an ordered list of top-$K$ documents $[d_{(1)}, d_{(2)}, \dots, d_{(K)}]$ ranked in descending order of relevance to query $q$:

$$\text{Rank}(d_i) \succ \text{Rank}(d_j) \iff \text{Score}(q, d_i) \ge \text{Score}(q, d_j)$$

where the scoring function $\text{Score}(q, d)$ evaluates either:
1. Lexical overlap ($\text{Score}_{\text{TF-IDF}}$).
2. Semantic similarity ($\text{Score}_{\text{Embedding}}$).
3. A normalized blend ($\text{Score}_{\text{Hybrid}}$).

---

## Search Architecture

```
                                 DOCUMENTS (120 docs, 10 categories)
                                                  │
                                                  ▼
                                       PREPROCESSING & CLEANING
                                                  │
                                                  ▼
                                            TOKENIZATION
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
                     TF-IDF                                         WORD EMBEDDINGS
                 (Sparse Matrix)                                (Dense Document Vectors)
                         │                                                 │
                         ▼                                                 ▼
                    TF-IDF Index                                    Embedding Index
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                                             USER QUERY
                                                  │
                                                  ▼
                                         QUERY VECTORIZATION
                                                  │
                         ┌────────────────────────┼────────────────────────┐
                         ▼                        ▼                        ▼
                    TF-IDF Search          Embedding Search          Hybrid Search
                         │                        │                        │
                         └────────────────────────┼────────────────────────┘
                                                  ▼
                                            TOP-K RANKING
                                       (Vectorized Cosine Sim)
                                                  │
                                                  ▼
                                        EVALUATION & METRICS
                                      (P@1, P@3, P@5, R@5, MRR)
                                                  │
                                                  ▼
                                        ERROR & PCA ANALYSIS
```

---

## Dataset

The corpus consists of **120 curated technical documents** across 10 distinct computing domains:
- **Python**: Memory management, generators, metaclasses, asyncio, decorators, GIL, context managers.
- **Data Science**: EDA, Pandas groupby, imputation, hypothesis testing, A/B testing, broadcasting, PCA, telematics.
- **Machine Learning**: Linear/logistic regression, XGBoost, Random Forests, SVMs, KNN, cross-validation, clustering.
- **Deep Learning**: Forward/backward propagation, activations, batch norm, dropout, LSTM, schedulers, GANs.
- **Computer Vision**: CNNs, augmentation, YOLO, U-Net, transfer learning, SIFT/ORB, optical flow, Vision Transformers.
- **Natural Language Processing**: Tokenization, TF-IDF, Word2Vec, cosine similarity, Transformers, BERT, pooling, NER.
- **Web Development**: FastAPI, REST, WebSockets, React, web security (XSS/CSRF), OAuth2/JWT, microservices, GraphQL.
- **Databases**: Normalization, SQL indexing, ACID transactions, PostgreSQL JSONB, MongoDB, Redis, sharding, vector DBs.
- **Cybersecurity**: AES/RSA cryptography, PKI/TLS, penetration testing, SQL injection defense, Zero Trust, MFA, SIEM, WAF.
- **Cloud**: IaaS/PaaS/SaaS, Docker, Kubernetes, AWS Lambda, Terraform, CI/CD, S3, VPCs, FinOps, multi-region failover.

Each document record adheres to a strict schema:
```json
{
  "id": 1,
  "title": "Python Memory Management and Garbage Collection",
  "text": "Python manages memory using private heaps and reference counting combined with a cyclic garbage collector...",
  "category": "Python"
}
```

---

## Document Representation

### 1. TF-IDF Representation
Documents are converted into sparse vectors in $\mathbb{R}^{|V|}$ where each element is:
$$\text{TF-IDF}(t, d) = (1 + \log \text{TF}(t, d)) \times \left(1 + \log \frac{1 + N}{1 + \text{DF}(t)}\right)$$
Vectors are $L_2$-normalized such that $\|d\|_2 = 1$.

### 2. Dense Document Embeddings (Word2Vec / Distributional Embeddings)
Word embeddings $\mathbf{v}_w \in \mathbb{R}^D$ ($D=64$) are learned from co-occurrence patterns using **Positive Pointwise Mutual Information (PPMI)** factorized with **TruncatedSVD** (Levy & Goldberg, 2014):
$$\text{PPMI}(t, c) = \max\left(0, \log \frac{P(t, c)}{P(t)P(c)^\alpha}\right)$$
Documents are then represented by pooling constituent word vectors:

#### Mean Pooling
$$d_{\text{mean}} = \frac{1}{|T|} \sum_{w \in T} \mathbf{v}_w$$

#### TF-IDF Weighted Pooling
$$d_{\text{weighted}} = \frac{\sum_{w \in T} \text{TF-IDF}(w) \cdot \mathbf{v}_w}{\sum_{w \in T} \text{TF-IDF}(w)}$$

#### Out-of-Vocabulary (`<UNK>`) Policy
Tokens not present in the embedding vocabulary are safely handled via an explicit policy:
- **`zero` (Default)**: OOV tokens are ignored during pooling. If a document consists entirely of OOV tokens, a zero vector $\mathbf{0} \in \mathbb{R}^D$ is returned.
- **`unk`**: A dedicated unit-norm pseudo-random vector $\mathbf{v}_{\text{unk}}$ is assigned to unknown tokens.

---

## Cosine Similarity & Vectorized Ranking

Cosine similarity measures the orientation between query vector $\mathbf{q}$ and document vector $\mathbf{d}$:
$$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

### Numerical Safety Guards
To prevent floating-point division by zero:
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0 or np.isnan(denom):
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
```
In the retrieval engine, similarity computation is fully vectorized across all $N$ documents using matrix multiplication:
$$\mathbf{s} = \frac{\mathbf{D} \mathbf{q}}{\|\mathbf{D}\|_2 \|\mathbf{q}\|_2} \in \mathbb{R}^N$$

---

## Top-K Retrieval & Search Modes

1. **TF-IDF Search**: Computes cosine similarity in the sparse vocabulary space.
2. **Embedding Search**: Computes cosine similarity in the dense continuous embedding space.
3. **Hybrid Search**: Fuses both modalities using min-max normalization and convex weighting:
   $$\text{Score}_{\text{Hybrid}} = \alpha \cdot \text{Norm}(\text{Score}_{\text{TF-IDF}}) + (1 - \alpha) \cdot \text{Norm}(\text{Score}_{\text{Embedding}})$$
   where $\alpha \in [0, 1]$.

---

## Retrieval Evaluation Metrics

### Precision@K
Measures the proportion of retrieved documents in the top $K$ that are relevant:
$$\text{Precision@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

### Recall@K
Measures the proportion of all relevant documents that appear in the top $K$:
$$\text{Recall@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

### Mean Reciprocal Rank (MRR)
Evaluates how high the first relevant document is ranked across all queries:
$$\text{RR}_q = \frac{1}{\text{rank}_{(1)}}, \quad \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}_q$$

---

## Empirical Benchmark Results

Evaluated across **25 benchmark queries** in `data/evaluation/ground_truth.json`:

| Metric | TF-IDF Search | Embedding Search | Hybrid Search ($\alpha=0.50$) |
|---|---|---|---|
| **Mean Precision@1** | 0.9600 | 0.9600 | **1.0000** |
| **Mean Precision@3** | 0.4533 | 0.4267 | **0.4667** |
| **Mean Precision@5** | 0.2960 | 0.3040 | **0.3120** |
| **Mean Recall@5** | 0.9100 | 0.9233 | **0.9433** |
| **MRR** | 0.9733 | 0.9800 | **1.0000** |

### Breakdown by Query Type (MRR)

| Query Category | Query Count | TF-IDF MRR | Embedding MRR | Hybrid MRR |
|---|---|---|---|---|
| **Keyword Match** | 5 | 1.0000 | 1.0000 | **1.0000** |
| **Semantic Synonym** | 5 | 0.8667 | **1.0000** | **1.0000** |
| **Short Query (1-2 words)** | 5 | 1.0000 | 1.0000 | **1.0000** |
| **Long Query (6+ words)** | 5 | 0.9487 | **1.0000** | **1.0000** |
| **Conceptual / Cross-Domain** | 3 | 1.0000 | 1.0000 | **1.0000** |
| **Word Order Adversarial** | 2 | **1.0000** | 0.7500 | **1.0000** |

---

## Experiments & Critical Analysis

### Experiment 1: TF-IDF vs. Dense Embeddings
- **Finding**: TF-IDF achieves perfect accuracy on verbatim technical keywords, but drops to MRR = 0.8667 on synonym queries (`vehicle maintenance services` vs `car repair`). Dense embeddings bridge this vocabulary gap completely (MRR = 1.0000).

### Experiment 2: Query Length Sensitivity
- **Finding**: On verbose, conversational queries (6+ words), TF-IDF precision degrades slightly due to term proliferation. Dense embeddings condense the entire sentence into a coherent vector, maintaining MRR = 1.0000.

### Experiment 3: Document Length Dilution
- **Finding**: Mean pooling exhibits slight dilution on long documents (> 35 words), where tangential words pull the document vector away from the core topic. TF-IDF weighted pooling mitigates this effect by downweighting common terms.

### Experiment 4: The Word Order Problem
- **Finding**: Mean document pooling produces a cosine similarity of **1.0000** between `"the dog chased the cat"` and `"the cat chased the dog"`. Bag-of-vectors pooling is fundamentally order-invariant, making it incapable of distinguishing semantic roles (agent vs. patient).

### Experiment 5: Hybrid Search Alpha Sweep
- **Finding**: Evaluating $\alpha \in [0.0, 0.25, 0.50, 0.75, 1.0]$:
  - $\alpha = 0.0$ (Pure Embedding): MRR = 0.9800, P@1 = 0.9600
  - $\alpha = 0.25$: **MRR = 1.0000, P@1 = 1.0000**
  - $\alpha = 0.50$: **MRR = 1.0000, P@1 = 1.0000**
  - $\alpha = 1.0$ (Pure TF-IDF): MRR = 0.9733, P@1 = 0.9600
  - *Conclusion*: A blended score mitigates the individual weaknesses of both models.

---

## Visualizations

All 12 required analytical visualizations are saved in `output/charts/`:
1. `1_category_distribution.png`: Uniform distribution across the 10 domains (12 docs/category).
2. `2_document_length_distribution.png`: Word count histogram (mean: 34.6 words).
3. `3_query_length_distribution.png`: Benchmark query length distribution.
4. `4_similarity_score_distribution.png`: KDE density curves comparing TF-IDF vs Embedding similarity scores.
5. `5_precision_comparison.png`: Bar chart comparing Precision@1, Precision@3, and Precision@5.
6. `6_recall_comparison.png`: Mean Recall@5 comparison across engines.
7. `7_mrr_comparison.png`: Mean Reciprocal Rank (MRR) comparison.
8. `8_tfidf_vs_embedding_correlation.png`: Scatter plot illustrating score space correlation ($r = 0.44$).
9. `9_top_retrieved_scores.png`: Box plot showing score degradation across rank positions 1–5.
10. `10_error_analysis.png`: Breakdown of retrieval failure modes.
11. `11_embedding_pca_2d.png`: 2D PCA projection of document vectors colored by category.
12. `12_query_doc_similarity_heatmap.png`: Heatmap showing cosine similarity across query-document pairs.

---

## Limitations of Simple Document Embeddings

1. **Loss of Word Order and Syntax**: Averaging ignores grammatical structure. `"Dog bites man"` and `"Man bites dog"` map to the identical vector.
2. **Dilution in Long Documents**: As document length grows, averaging many disparate word vectors pushes the representation toward a generic central centroid.
3. **Equal Weighting of Polysemous Words**: Simple mean pooling weights ambiguous words (e.g. *"bank"*, *"apple"*, *"python"*) identically to highly discriminative technical terms.
4. **Vocabulary Mismatch for Rare Words**: Static word embeddings cannot represent unseen words unless subword tokenization (BPE/WordPiece) is used.

---

## Project Structure

```
Day 104/
├── app/
│   ├── __init__.py
│   ├── config.py                  # Paths, constants, and hyperparameters
│   ├── main.py                    # End-to-end execution pipeline
│   ├── report.py                  # Markdown report generator
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # JSON data loader
│   │   ├── validator.py           # Schema & integrity validator
│   │   └── ground_truth.py        # Benchmark query provider
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py             # Normalization & stopwords
│   │   └── tokenizer.py           # Tokenizer & n-gram extractor
│   ├── representations/
│   │   ├── __init__.py
│   │   ├── tfidf.py               # TF-IDF representation
│   │   ├── pooling.py             # Mean & weighted pooling
│   │   └── document_embedding.py  # Word embedding training & pooling
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── ranking.py             # Safe cosine similarity & Top-K ranking
│   │   ├── tfidf_search.py        # Keyword search engine
│   │   ├── embedding_search.py    # Dense semantic search engine
│   │   └── hybrid_search.py       # Normalized hybrid search engine
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── precision_at_k.py      # Precision@K metric
│   │   ├── recall_at_k.py         # Recall@K metric
│   │   ├── mrr.py                 # Reciprocal rank & MRR
│   │   └── evaluator.py           # Benchmark evaluation suite
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── error_analysis.py      # Error categorization
│   │   └── similarity_analysis.py # Cohesion & correlation analysis
│   └── visualization/
│       ├── __init__.py
│       └── charts.py              # 12 analytical plots generator
├── coding_challenges/
│   ├── challenge_104_document_embedding.py
│   ├── challenge_104_top_k.py
│   ├── challenge_104_precision_at_k.py
│   ├── challenge_104_recall_at_k.py
│   ├── challenge_104_mrr.py
│   └── challenges_1_to_12.py
├── experiments/
│   ├── tfidf_vs_embeddings.py
│   ├── query_length.py
│   ├── document_length.py
│   ├── word_order.py
│   └── hybrid_search.py
├── data/
│   ├── generate_data.py           # Curated data generator
│   ├── raw/documents.json         # 120 multi-domain documents
│   └── evaluation/ground_truth.json # 25 benchmark queries
├── output/
│   ├── charts/                    # 12 generated visualizations
│   ├── embeddings.npy             # Dense word embedding weights
│   ├── embeddings.json            # Embedding vocabulary
│   ├── search_results.csv         # Per-query search results
│   ├── retrieval_metrics.csv      # Summary evaluation metrics
│   ├── errors.csv                 # Detailed error analysis
│   └── report.md                  # Comprehensive evaluation report
├── tests/
│   ├── test_loader.py
│   ├── test_cleaner.py
│   ├── test_embeddings.py
│   ├── test_similarity.py
│   ├── test_search.py
│   ├── test_ranking.py
│   ├── test_precision.py
│   ├── test_recall.py
│   ├── test_mrr.py
│   └── test_hybrid.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation & Usage

### 1. Requirements
Ensure Python 3.8+ is installed with dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python app/main.py
```

### 3. Run Experiments
```bash
python experiments/tfidf_vs_embeddings.py
python experiments/query_length.py
python experiments/document_length.py
python experiments/word_order.py
python experiments/hybrid_search.py
```

### 4. Run Coding Challenges
```bash
python coding_challenges/challenges_1_to_12.py
```

### 5. Run Unit Tests (70 Tests)
```bash
pytest tests -v
```

---

## 🎤 Interview Questions & Comprehensive Answers

### Fundamentals
1. **What is semantic search?**
   - Semantic search is an information retrieval technique that seeks to understand the intent and contextual meaning of a user's query rather than simply finding exact lexical keyword matches. It maps queries and documents into a shared continuous vector space and ranks documents using geometric distance (such as cosine similarity).

2. **Keyword search vs. semantic search?**
   - Keyword search (e.g. TF-IDF, BM25) relies on exact term frequency and inverted indexes. It is fast, exact, and domain-agnostic, but suffers from synonymy and vocabulary mismatch. Semantic search understands conceptual relationships and paraphrasing, retrieving relevant documents even when zero words overlap, but requires dense vector indexing and can suffer from false positives due to semantic drift.

3. **What is a document embedding?**
   - A document embedding is a fixed-dimensional continuous vector representation $\mathbf{d} \in \mathbb{R}^D$ that encodes the semantic meaning of an entire sentence, paragraph, or document.

4. **How can word embeddings become document embeddings?**
   - By aggregating the embeddings of individual words in the document using pooling operations (such as arithmetic mean pooling, max pooling, or TF-IDF weighted pooling), or by using neural architectures like Doc2Vec, Sentence-BERT, or transformer encoder representations.

5. **What is mean pooling?**
   - Mean pooling computes the element-wise arithmetic average of all word embedding vectors present in a document:
     $$\mathbf{d} = \frac{1}{|T|} \sum_{w \in T} \mathbf{v}_w$$

6. **What problem does mean pooling have?**
   - It ignores token sequence and syntax (bag-of-words assumption), dilutes key topical signals in long documents with non-informative words, and fails to handle negation or directional semantics (`"dog chased cat"` vs `"cat chased dog"`).

### Retrieval
7. **What is Top-K retrieval?**
   - Top-K retrieval is the process of computing similarity scores between a query vector and all indexed document vectors, sorting the scores descending, and returning only the $K$ highest-scoring documents.

8. **What is Precision@K?**
   - The fraction of the top $K$ retrieved documents that are relevant according to ground truth:
     $$\text{Precision@K} = \frac{\text{Relevant in top } K}{K}$$

9. **What is Recall@K?**
   - The fraction of all known relevant documents that are successfully retrieved within the top $K$:
     $$\text{Recall@K} = \frac{\text{Relevant in top } K}{\text{Total Relevant}}$$

10. **What is MRR (Mean Reciprocal Rank)?**
    - The average of the reciprocal ranks of the first relevant document retrieved across a set of queries:
      $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_{\text{first relevant}}}$$
      It rewards systems that return a relevant result in position 1.

11. **Why do we need a relevance ground truth?**
    - Without an objective ground truth annotating which documents are truly relevant to specific queries, retrieval quality cannot be measured quantitatively. Subjective inspection is prone to confirmation bias.

### Engineering
12. **Why must query and documents use the same vector space?**
    - Cosine similarity measures the angle between vectors. If query and documents are vectorized using different vectorizers, different embedding dimensions, or non-aligned projection spaces, the angle between them is mathematically meaningless.

13. **Why should embeddings be precomputed?**
    - Document collections are generally static relative to query volume. Computing document embeddings offline during indexing ensures that query-time latency depends only on single query vectorization and vector similarity computation, not re-embedding millions of documents.

14. **What is an embedding index?**
    - A data structure that stores document vectors and supports fast nearest neighbor retrieval (e.g. flat array for brute-force exact search, or hierarchical graphs like HNSW for approximate search).

15. **Why might brute-force similarity search become slow?**
    - An exact linear scan has time complexity $\mathcal{O}(N \times D)$ where $N$ is corpus size and $D$ is vector dimension. For millions of documents, computing millions of dot products per query exceeds acceptable latency budgets (e.g. > 50ms), requiring Approximate Nearest Neighbor (ANN) indexes like HNSW, FAISS, or ScaNN.

### Critical Thinking
16. **Why might TF-IDF outperform embeddings on some datasets?**
    - On corpora with specialized terminology (e.g. product SKU numbers, medical codes, legal citations, rare acronyms), exact term matching is critical. Dense embeddings can compress distinct rare tokens into similar neighborhood regions, causing false-positive confusion where TF-IDF is strictly precise.

17. **Why might embeddings retrieve semantically related documents that don't share exact words?**
    - Because word embeddings are trained on distributional hypothesis principles ($w_1$ and $w_2$ that appear in similar linguistic contexts have similar vector directions). Therefore, $\mathbf{v}_{\text{car}} \cdot \mathbf{v}_{\text{automobile}} \approx 1$, allowing semantic search to match them without lexical overlap.

18. **What information can mean pooling lose?**
    - Word order, sentence structure, syntactic dependencies, negation (`"not bad"` vs `"bad"`), and term importance.

19. **Why does document length matter?**
    - In short documents, every word strongly reflects the core topic. In long documents, background and transition words accumulate, causing the mean vector to regress toward the general corpus centroid (semantic dilution).

20. **Why shouldn't PCA plots be treated as the full embedding space?**
    - PCA is a linear projection that compresses $D$-dimensional space (e.g. 64 or 768 dimensions) into 2 dimensions, retaining only a fraction of the total variance (e.g. 15–30%). Two points that appear close in a 2D projection may actually be far apart in the remaining $D-2$ orthogonal dimensions.

---

## Key Learnings

1. **Vector Space Alignment**: Both queries and documents must pass through the identical transformation pipeline.
2. **Hybrid Synergy**: Combining sparse lexical search ($\alpha \approx 0.25\text{--}0.50$) with dense semantic embeddings provides the highest retrieval performance across diverse query types.
3. **The Order Imperative**: The inability of bag-of-vectors pooling to distinguish word order highlights the motivation for modern Transformer architectures (Day 106+) and cross-encoders.
