"""
Generates synthetic corpus of 30 technical documents and 60+ benchmark evaluation queries.
Covers Python, DBMS, Machine Learning, Deep Learning, Systems, and Networking.
"""
from pathlib import Path
import json

DOCUMENTS = [
    # Category: Python
    {
        "filename": "python_list_comprehension.md",
        "category": "python",
        "title": "Python List Comprehensions and Generator Expressions",
        "tags": ["python", "syntax", "comprehension", "iterators", "performance"],
        "content": """# Python List Comprehensions and Generator Expressions

Python list comprehension provides a concise and readable syntax for creating new lists from existing iterables.
The canonical syntax is `[expression for item in iterable if condition]`.
List comprehensions evaluate eagerly and construct the full list in memory immediately.

In contrast, generator expressions use parentheses `(expression for item in iterable if condition)` and produce iterators that yield items lazily on demand.
Generator expressions have an O(1) memory footprint regardless of stream size, making them ideal for unbounded or large datasets.

Best practices recommend avoiding nested list comprehensions deeper than two levels to preserve code readability.
For simple map-and-filter workflows, list comprehensions execute significantly faster than manual `for` loops appending to an empty list because the bytecode executes at C-speed in CPython."""
    },
    {
        "filename": "python_memory_management.md",
        "category": "python",
        "title": "Python Memory Management and Garbage Collection",
        "tags": ["python", "memory", "garbage collection", "reference count", "leak"],
        "content": """# Python Memory Management and Garbage Collection

Python manages memory using an internal private heap managed by the PyMalloc allocator for small allocations up to 512 bytes.
The primary memory reclamation mechanism is reference counting. Every Python object maintains an `ob_refcnt` header.
When `ob_refcnt` drops to zero, the memory occupied by the object is deallocated immediately.

To handle cyclic references that reference counting cannot resolve, Python incorporates a generational cyclical garbage collector (GC).
The GC organizes objects into three generations (Gen 0, Gen 1, Gen 2) based on object survival across collection cycles.
Gen 0 is inspected frequently, whereas Gen 2 contains long-lived objects.
A common source of memory leaks in Python is retaining references in global dictionaries, unclosed file descriptors, or cycles containing custom `__del__` implementations prior to Python 3.4."""
    },
    {
        "filename": "python_gil_concurrency.md",
        "category": "python",
        "title": "Python Global Interpreter Lock (GIL) and Concurrency",
        "tags": ["python", "gil", "concurrency", "multithreading", "multiprocessing"],
        "content": """# Python Global Interpreter Lock (GIL) and Concurrency

The Global Interpreter Lock (GIL) is a mutex in CPython that prevents multiple native OS threads from executing Python bytecode simultaneously.
The GIL exists primarily because CPython's memory management is not thread-safe without coarse-grained locking.

For CPU-bound tasks, multithreading in Python does not achieve true parallelism across multiple CPU cores due to GIL contention.
In contrast, CPU-bound tasks benefit from `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`, which instantiate separate Python processes with independent GILs and memory spaces.
For I/O-bound operations (such as network requests or disk reads), Python threads release the GIL while waiting on the kernel, enabling high concurrency via threading or `asyncio`."""
    },
    {
        "filename": "python_decorators_closures.md",
        "category": "python",
        "title": "Python Decorators and Lexical Closures",
        "tags": ["python", "decorators", "closures", "functional", "functions"],
        "content": """# Python Decorators and Lexical Closures

In Python, functions are first-class citizens that can be passed as arguments, returned from other functions, and bound to variables.
A closure occurs when a nested inner function retains references to variables from its enclosing lexical scope even after the outer function has completed execution.

Decorators leverage closures to wrap functions or methods, modifying their behavior dynamically without altering the original source code.
Using `functools.wraps` preserves the wrapped function's metadata, including docstring, name, and parameter signatures."""
    },
    {
        "filename": "python_generators_iterators.md",
        "category": "python",
        "title": "Python Iterators, Iterables, and Generators",
        "tags": ["python", "generators", "yield", "iterators", "protocol"],
        "content": """# Python Iterators, Iterables, and Generators

An iterable is any Python object capable of returning its members one at a time, implementing the `__iter__()` method.
An iterator is an object representing a stream of data that implements both `__iter__()` and `__next__()` protocols.
When no further items remain in the sequence, `__next__()` raises the `StopIteration` exception.

Generators provide a simple way to implement iterators using the `yield` keyword.
When a generator function calls `yield`, its state, local variables, and instruction pointer are suspended and returned to the caller."""
    },

    # Category: DBMS / SQL
    {
        "filename": "dbms_sql_inner_join.md",
        "category": "dbms",
        "title": "SQL INNER JOIN Principles and Query Execution",
        "tags": ["dbms", "sql", "inner join", "relational", "query", "database"],
        "content": """# SQL INNER JOIN Principles and Query Execution

The SQL INNER JOIN clause selects records that have matching values in both joined tables based on a specified join predicate.
The standard syntax is:
`SELECT columns FROM table_a INNER JOIN table_b ON table_a.foreign_key = table_b.primary_key;`

If a row in Table A does not match any row in Table B under the ON predicate, it is omitted from the resulting relation.
Relational query optimizers evaluate join strategies using algorithms such as Nested Loop Join, Hash Join, or Sort-Merge Join depending on table statistics and available indexes.
For large datasets with equi-join conditions, Hash Joins typically provide O(M + N) performance when an in-memory hash table fits in `work_mem`."""
    },
    {
        "filename": "dbms_sqlstate_23505_errors.md",
        "category": "dbms",
        "title": "Handling SQLSTATE 23505 Unique Constraint Violations",
        "tags": ["dbms", "sql", "sqlstate 23505", "unique constraint", "integrity", "postgres"],
        "content": """# Handling SQLSTATE 23505 Unique Constraint Violations

In relational databases (especially PostgreSQL and ANSI SQL compliant engines), error code SQLSTATE 23505 designates a unique constraint violation.
This error is triggered when an `INSERT` or `UPDATE` statement attempts to write a duplicate value into a column or composite columns constrained by a `UNIQUE` index or `PRIMARY KEY`.

To handle SQLSTATE 23505 gracefully without rolling back an entire application transaction, developers utilize `INSERT ... ON CONFLICT (column_name) DO UPDATE` or `DO NOTHING` in PostgreSQL (upsert pattern).
In distributed databases, mitigating unique constraint collisions often requires distributed ID generation schemes such as Snowflake IDs or ULIDs."""
    },
    {
        "filename": "dbms_acid_transactions.md",
        "category": "dbms",
        "title": "ACID Properties and Transaction Isolation Levels",
        "tags": ["dbms", "acid", "transactions", "isolation", "concurrency"],
        "content": """# ACID Properties and Transaction Isolation Levels

Database transactions adhere to ACID properties: Atomicity (all-or-nothing execution), Consistency (maintaining valid schema invariants), Isolation (concurrent transactions execute without interference), and Durability (committed changes survive system crashes).

ANSI SQL defines four isolation levels: Read Uncommitted, Read Committed, Repeatable Read, and Serializable.
Higher isolation levels prevent concurrency anomalies such as Dirty Reads, Non-Repeatable Reads, and Phantom Reads, but may increase lock contention and serialization failures.
Modern relational databases like PostgreSQL utilize Multi-Version Concurrency Control (MVCC) to ensure readers do not block writers and writers do not block readers."""
    },
    {
        "filename": "dbms_indexing_btree_hash.md",
        "category": "dbms",
        "title": "Database Indexing: B-Tree vs Hash Indexes",
        "tags": ["dbms", "indexing", "b-tree", "hash", "performance"],
        "content": """# Database Indexing: B-Tree vs Hash Indexes

B-Tree indexes are balanced search trees where leaf nodes are sorted and linked sequentially.
They support equality lookups `(=)` as well as range scans `(<, <=, >, >=, BETWEEN)` in O(log N) time.
B-Trees are the default index structure in PostgreSQL, MySQL, and SQLite.

Hash indexes build an in-memory hash table of keys pointing to row IDs.
They offer O(1) average lookup time for exact equality searches but cannot accelerate range queries, sorting (`ORDER BY`), or partial prefix searches."""
    },
    {
        "filename": "dbms_deadlock_detection.md",
        "category": "dbms",
        "title": "Deadlock Detection and Prevention in Relational Engines",
        "tags": ["dbms", "deadlock", "concurrency", "locking", "transactions"],
        "content": """# Deadlock Detection and Prevention in Relational Engines

A deadlock occurs when two or more transactions maintain exclusive locks on resources while waiting to acquire locks held by each other, forming a cyclic dependency graph.
Neither transaction can proceed without external intervention.

Relational database engines resolve deadlocks using wait-for-graph cycle detection algorithms.
When a cycle is detected, the database engine aborts one of the transactions (the deadlock victim) and raises an error, allowing the other transaction to proceed.
Developers can prevent deadlocks by always acquiring table locks in a deterministic, global alphabetical order and keeping transactions short."""
    },

    # Category: Machine Learning
    {
        "filename": "ml_gradient_descent_learning_rate.md",
        "category": "machine_learning",
        "title": "Gradient Descent Optimization and Learning Rate Tuning",
        "tags": ["machine_learning", "gradient descent", "learning rate", "optimization", "sgd"],
        "content": """# Gradient Descent Optimization and Learning Rate Tuning

Gradient descent is a first-order iterative optimization algorithm used to minimize the cost function $J(\\theta)$ by updating model parameters in the direction of the negative gradient.
The fundamental parameter update formula is:
$$\\theta_{t+1} = \\theta_t - \\eta \\nabla J(\\theta_t)$$
where $\\eta$ is the learning rate.

The learning rate $\\eta$ is the single most critical hyperparameter in optimization:
1. Too small $\\eta$: Convergence is excruciatingly slow, and optimization risks being trapped in shallow sub-optimal local minima.
2. Too large $\\eta$: Updates overshoot the minimum, causing loss oscillation or numerical divergence ($NaN$).

Adaptive optimizers such as Adam, RMSprop, and AdaGrad scale individual coordinate learning rates using historical first and second moment statistics of the gradients.
Learning rate schedules (such as cosine annealing with warmup) systematically decay $\\eta$ across epochs to stabilize late-stage convergence."""
    },
    {
        "filename": "ml_overfitting_regularization.md",
        "category": "machine_learning",
        "title": "Mitigating Overfitting in Machine Learning Models",
        "tags": ["machine_learning", "overfitting", "regularization", "dropout", "early stopping"],
        "content": """# Mitigating Overfitting in Machine Learning Models

Overfitting occurs when a statistical model memorizes noise and idiosyncrasies in the training set rather than learning underlying generalizable patterns, leading to low training loss but high validation loss.

Core techniques to reduce overfitting include:
1. L2 Regularization (Weight Decay / Ridge): Penalizes large weight magnitudes by adding $\\frac{\\lambda}{2} \\sum w_i^2$ to the loss function, smoothing the decision boundary.
2. L1 Regularization (Lasso): Adds $\\lambda \\sum |w_i|$ to the objective, inducing parameter sparsity for automatic feature selection.
3. Early Stopping: Halts training when validation loss stops improving for a specified patience window, preventing the model from fitting training noise.
4. Data Augmentation: Synthetically expands the dataset via rotations, cropping, or noise injection to improve invariant representations.
5. Dropout: Randomly deactivates neurons during forward passes with probability $p$, preventing complex co-adaptations."""
    },
    {
        "filename": "ml_bias_variance_tradeoff.md",
        "category": "machine_learning",
        "title": "The Bias-Variance Tradeoff in Supervised Learning",
        "tags": ["machine_learning", "bias", "variance", "tradeoff", "generalization"],
        "content": r"""# The Bias-Variance Tradeoff in Supervised Learning

The expected generalization error of a machine learning model decomposes into three distinct components: Bias error, Variance error, and Irreducible error ($\sigma^2$).
Bias represents error stemming from erroneous simplifying assumptions in the learning algorithm; high bias leads to underfitting.
Variance measures sensitivity to small fluctuations in the training dataset; high variance leads to overfitting.

Ensemble methods directly address this tradeoff:
Bagging (Bootstrap Aggregating, such as Random Forests) reduces variance without increasing bias by averaging predictions from multiple de-correlated base estimators.
Boosting (such as XGBoost or LightGBM) sequentially fits weak learners to residual errors, primarily reducing bias."""
    },
    {
        "filename": "ml_feature_scaling_normalization.md",
        "category": "machine_learning",
        "title": "Feature Scaling: Standardization vs Min-Max Normalization",
        "tags": ["machine_learning", "preprocessing", "scaling", "standardization", "normalization"],
        "content": r"""# Feature Scaling: Standardization vs Min-Max Normalization

Distance-based algorithms (such as KNN, SVMs, and K-Means) and gradient-based models are sensitive to differences in feature magnitudes.
Standardization (Z-score normalization) transforms features to have zero mean and unit variance: $z = (x - \mu) / \sigma$.
Min-Max normalization scales values into a fixed bounded interval $[0, 1]$: $x_{norm} = (x - x_{min}) / (x_{max} - x_{min})$.

Standardization is preferred when features exhibit Gaussian distributions or contain outliers, as Min-Max scaling compresses inliers into a tiny range when extreme outliers are present."""
    },
    {
        "filename": "ml_cross_validation_strategies.md",
        "category": "machine_learning",
        "title": "Cross-Validation and Data Leakage Prevention",
        "tags": ["machine_learning", "evaluation", "cross-validation", "leakage", "stratified"],
        "content": """# Cross-Validation and Data Leakage Prevention

K-Fold Cross-Validation splits training data into $K$ disjoint folds, training the model on $K-1$ folds and evaluating on the held-out fold iteratively.
Stratified K-Fold preserves class proportion across each partition, which is critical for imbalanced classification tasks.

Data leakage occurs when information from outside the training dataset influences model training.
To prevent leakage, all feature scalers, imputers, and dimensionality reduction projections must be fitted strictly on the training partition and only applied to validation/test folds."""
    },

    # Category: NLP & Information Retrieval
    {
        "filename": "nlp_tfidf_vectorization.md",
        "category": "nlp",
        "title": "TF-IDF Weighting and Vector Space Retrieval",
        "tags": ["nlp", "tf-idf", "retrieval", "vector space", "information retrieval"],
        "content": """# TF-IDF Weighting and Vector Space Retrieval

Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic reflecting how important a word is to a document in a collection.
The term frequency $TF(t, d)$ measures the raw or sublinear count of term $t$ in document $d$.
The inverse document frequency $IDF(t) = \\ln(1 + N / n_t)$ penalizes words that appear ubiquitously across all documents (such as 'the', 'is', 'and').

In the vector space model, documents and queries are represented as sparse high-dimensional vectors with TF-IDF weights.
Relevance is scored by computing the cosine similarity between the query vector and document vectors.
While TF-IDF is computationally efficient and interpretable, it suffers from vocabulary mismatch and cannot capture semantic synonyms."""
    },
    {
        "filename": "nlp_bm25_retrieval_ranking.md",
        "category": "nlp",
        "title": "BM25 Probabilistic Ranking Function Architecture",
        "tags": ["nlp", "bm25", "okapi", "retrieval", "search", "ranking"],
        "content": """# BM25 Probabilistic Ranking Function Architecture

Okapi BM25 is a non-linear ranking function derived from probabilistic relevance theory that improves upon traditional TF-IDF.
Its scoring function is expressed as:
$$\\operatorname{BM25}(D, Q) = \\sum_{q_i \\in Q} \\operatorname{IDF}(q_i) \\cdot \\frac{f(q_i, D)(k_1 + 1)}{f(q_i, D) + k_1(1 - b + b \\frac{|D|}{\\text{avgdl}})}$$

The parameter $k_1$ controls term frequency saturation; as term frequency increases, its marginal relevance score approaches an asymptotic ceiling.
The parameter $b$ controls document length normalization; when $b = 1$, long documents are heavily penalized for verbosity, while $b = 0$ disables length compensation entirely.
BM25 excels at exact keyword matching, error codes, and technical identifiers."""
    },
    {
        "filename": "nlp_dense_embeddings_semantic_search.md",
        "category": "nlp",
        "title": "Dense Embeddings and Semantic Vector Retrieval",
        "tags": ["nlp", "embeddings", "dense", "semantic search", "cosine"],
        "content": """# Dense Embeddings and Semantic Vector Retrieval

Dense vector representations map textual sequences into continuous low-dimensional geometric spaces (typically 64 to 1536 dimensions) where semantic similarity corresponds to vector proximity.
Models encode latent meaning beyond exact token matches, allowing queries like 'feline health' to match documents about 'cat medical care'.

Nearest neighbor search in dense vector spaces relies on cosine similarity:
$$\\text{Sim}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}$$
For web-scale search over billions of vectors, Approximate Nearest Neighbor (ANN) indexes such as HNSW (Hierarchical Navigable Small World) and IVF-PQ provide logarithmic query latency."""
    },
    {
        "filename": "nlp_reciprocal_rank_fusion_rrf.md",
        "category": "nlp",
        "title": "Reciprocal Rank Fusion (RRF) for Hybrid Search",
        "tags": ["nlp", "rrf", "hybrid search", "ranking", "fusion"],
        "content": """# Reciprocal Rank Fusion (RRF) for Hybrid Search

Reciprocal Rank Fusion (RRF) is an unsupervised rank aggregation method that combines multiple ranked result lists into a single consolidated ranking.
The RRF score for document $d$ across ranking systems $R$ is defined as:
$$\\operatorname{RRF}(d) = \\sum_{r \\in R} \\frac{1}{k_0 + \\operatorname{rank}_r(d)}$$
where $k_0$ is a smoothing constant (conventionally set to 60) and $\\operatorname{rank}_r(d)$ is the 1-based position of document $d$ in system $r$.

RRF is parameter-free, requires no score calibration or normalization across disparate scoring distributions (such as unbounded BM25 scores versus bounded cosine similarities), and reliably outperforms either individual retriever."""
    },
    {
        "filename": "nlp_maximal_marginal_relevance_mmr.md",
        "category": "nlp",
        "title": "Maximal Marginal Relevance (MMR) for Result Diversity",
        "tags": ["nlp", "mmr", "diversity", "novelty", "retrieval"],
        "content": """# Maximal Marginal Relevance (MMR) for Result Diversity

In retrieval systems, returning the top-K most similar documents often leads to extreme informational redundancy, where multiple chunks reiterate identical sentences.
Maximal Marginal Relevance (MMR) resolves this by balancing query relevance against novelty relative to already selected results:
$$\\operatorname{MMR} = \\arg\\max_{d_i \\in R \\setminus S} \\left[ \\lambda \\cdot \\operatorname{Sim}_1(d_i, Q) - (1 - \\lambda) \\max_{d_j \\in S} \\operatorname{Sim}_2(d_i, d_j) \\right]$$

Setting $\\lambda = 1.0$ prioritizes pure relevance (standard top-K ranking), whereas $\\lambda = 0.0$ promotes maximal diversity.
A balanced setting of $\\lambda = 0.7$ provides strong query relevance while suppressing duplicate content in LLM context windows."""
    },

    # Category: Systems & Networking
    {
        "filename": "systems_http_404_error.md",
        "category": "systems",
        "title": "HTTP 404 Error Status Code and Web Server Routing",
        "tags": ["systems", "networking", "http 404 error", "http status", "web", "rest"],
        "content": """# HTTP 404 Error Status Code and Web Server Routing

The HTTP 404 Not Found client error status response code indicates that the web server cannot locate the requested resource at the specified Uniform Resource Identifier (URI).
Unlike HTTP 401 Unauthorized or 403 Forbidden, a 404 indicates the endpoint or file path does not exist in the routing table.

In RESTful APIs, returning a 404 is appropriate when a requested entity ID does not exist in the underlying database.
In single-page applications (SPAs), web servers such as Nginx or Apache must rewrite missing asset paths to `index.html` (fallback routing) to enable client-side routers to handle internal navigation without returning 404 errors."""
    },
    {
        "filename": "systems_http_status_codes_overview.md",
        "category": "systems",
        "title": "HTTP Status Code Classes and Error Handling",
        "tags": ["systems", "networking", "http", "status codes", "api", "rest"],
        "content": """# HTTP Status Code Classes and Error Handling

HTTP response status codes are grouped into five categorical classes:
1. 1xx (Informational): Request received, continuing process (e.g., 101 Switching Protocols).
2. 2xx (Success): Action successfully received, understood, and accepted (e.g., 200 OK, 201 Created, 204 No Content).
3. 3xx (Redirection): Further action required to complete request (e.g., 301 Moved Permanently, 304 Not Modified).
4. 4xx (Client Error): Request contains bad syntax or cannot be fulfilled (e.g., 400 Bad Request, 403 Forbidden, 404 Not Found, 429 Too Many Requests).
5. 5xx (Server Error): Server failed to fulfill an apparently valid request (e.g., 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable)."""
    },
    {
        "filename": "systems_tcp_ip_three_way_handshake.md",
        "category": "systems",
        "title": "TCP Three-Way Handshake and Connection Teardown",
        "tags": ["systems", "networking", "tcp", "handshake", "protocols"],
        "content": """# TCP Three-Way Handshake and Connection Teardown

The Transmission Control Protocol (TCP) establishes reliable bidirectional connections using a three-way handshake:
1. SYN: The client generates an Initial Sequence Number (ISN) and sends a packet with the SYN flag set.
2. SYN-ACK: The server responds with its own SYN packet and acknowledges the client's ISN with an ACK (ISN + 1).
3. ACK: The client acknowledges the server's sequence number with an ACK packet. The connection enters the ESTABLISHED state.

Connection teardown uses a four-way handshake involving FIN and ACK packets from both endpoints, with the initiating client entering the TIME_WAIT state to ensure duplicate segments expire."""
    },
    {
        "filename": "systems_dns_resolution_lifecycle.md",
        "category": "systems",
        "title": "Domain Name System (DNS) Resolution Lifecycle",
        "tags": ["systems", "networking", "dns", "domain", "ip", "cache"],
        "content": """# Domain Name System (DNS) Resolution Lifecycle

DNS translates human-readable domain names (such as `example.com`) into numerical IP addresses.
When a client requests a domain, the resolution hierarchy traverses:
1. Local browser cache and operating system resolver cache.
2. Recursive DNS resolver (usually provided by an ISP or public resolver like 8.8.8.8).
3. Root nameservers (directing queries to TLD servers).
4. Top-Level Domain (TLD) nameservers (.com, .org).
5. Authoritative nameservers for the specific domain, which return the canonical A or AAAA records.
DNS caching at every level leverages Time-to-Live (TTL) headers to minimize query latency."""
    },
    {
        "filename": "systems_rest_api_idempotency.md",
        "category": "systems",
        "title": "REST API Idempotency and HTTP Method Semantics",
        "tags": ["systems", "rest", "api", "idempotency", "http"],
        "content": """# REST API Idempotency and HTTP Method Semantics

An HTTP method is considered idempotent if executing identical requests multiple times produces the exact same server-side state as executing it a single time.
`GET`, `HEAD`, `OPTIONS`, `PUT`, and `DELETE` are defined as idempotent methods under the HTTP specification.
`POST` and `PATCH` are typically non-idempotent because repeated submissions can generate duplicate entities or side-effects.

To make `POST` transactions idempotent in payment and billing APIs, clients transmit unique `Idempotency-Key` headers stored in an in-memory cache like Redis."""
    },

    # Category: Deep Learning
    {
        "filename": "dl_transformer_attention_mechanism.md",
        "category": "deep_learning",
        "title": "Transformer Scaled Dot-Product Attention Mechanisms",
        "tags": ["deep_learning", "transformer", "attention", "neural", "nlp"],
        "content": """# Transformer Scaled Dot-Product Attention Mechanisms

The core innovation of the Transformer architecture is Scaled Dot-Product Attention, computed as:
$$\\operatorname{Attention}(Q, K, V) = \\operatorname{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
where $Q$ represents Queries, $K$ represents Keys, $V$ represents Values, and $d_k$ is key dimensionality.

Scaling by $1 / \\sqrt{d_k}$ prevents the dot products from growing excessively large for large projection dimensions, which would otherwise push the softmax function into regions with vanishingly small gradients.
Multi-Head Attention projects $Q, K, V$ into $h$ distinct representation subspaces, enabling the model to attend to information from different representation positions simultaneously."""
    },
    {
        "filename": "dl_cross_entropy_loss_function.md",
        "category": "deep_learning",
        "title": "Cross-Entropy Loss in Multi-Class Classification",
        "tags": ["deep_learning", "loss", "cross entropy", "softmax", "optimization"],
        "content": """# Cross-Entropy Loss in Multi-Class Classification

Cross-entropy loss measures the divergence between true label probability distribution $y$ and predicted probability distribution $\\hat{y}$.
For multi-class classification with $C$ classes, categorical cross-entropy loss is defined as:
$$L = -\\sum_{c=1}^C y_c \\log(\\hat{y}_c)$$
where $\\hat{y}_c = \\frac{e^{z_c}}{\\sum_{j=1}^C e^{z_j}}$ is the softmax activation over logits $z$.

Combining softmax and cross-entropy yields a remarkably elegant and numerically stable gradient:
$$\\frac{\\partial L}{\\partial z_i} = \\hat{y}_i - y_i$$
which directly equates to the prediction error on logit $i$."""
    },
    {
        "filename": "dl_batch_normalization_layer.md",
        "category": "deep_learning",
        "title": "Batch Normalization Principles and Covariate Shift",
        "tags": ["deep_learning", "batch normalization", "neural networks", "training"],
        "content": """# Batch Normalization Principles and Covariate Shift

Batch Normalization (BatchNorm) stabilizes deep neural network training by normalizing layer activations across mini-batches during forward propagation.
Given mini-batch activations $x$, BatchNorm calculates batch mean $\\mu_B$ and batch variance $\\sigma_B^2$, normalizes the inputs, and applies learnable scale $(\\gamma)$ and shift $(\\beta)$ parameters:
$$\\hat{x} = \\frac{x - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}, \\quad y = \\gamma \\hat{x} + \\beta$$

BatchNorm smooths the optimization landscape, enables significantly higher learning rates, reduces sensitivity to parameter initialization, and acts as a mild regularizer."""
    },
    {
        "filename": "dl_backpropagation_chain_rule.md",
        "category": "deep_learning",
        "title": "Backpropagation and Reverse-Mode Automatic Differentiation",
        "tags": ["deep_learning", "backpropagation", "autograd", "calculus", "neural"],
        "content": """# Backpropagation and Reverse-Mode Automatic Differentiation

Backpropagation computes partial derivatives of the scalar objective loss with respect to all network parameters using the multivariate chain rule of calculus.
During the forward pass, intermediate activations are computed and cached in the computational graph.
During the backward pass, adjoint sensitivities flow in reverse order from the output loss back to initial input weights.

Modern deep learning frameworks (such as PyTorch and JAX) implement reverse-mode automatic differentiation (Autograd), which calculates gradients in time proportional to a single forward pass."""
    },
    {
        "filename": "dl_cross_encoder_vs_bi_encoder.md",
        "category": "deep_learning",
        "title": "Cross-Encoders vs Bi-Encoders in Neural Reranking",
        "tags": ["deep_learning", "cross-encoder", "bi-encoder", "reranking", "information retrieval"],
        "content": """# Cross-Encoders vs Bi-Encoders in Neural Reranking

Information retrieval systems employ two contrasting neural architectures: Bi-Encoders and Cross-Encoders.
In a Bi-Encoder architecture, the query and document are encoded completely independently into dense vector embeddings $u$ and $v$.
Relevance is scored by computing their dot product or cosine similarity.
Bi-encoders are fast and enable index pre-computation with MIPS/ANN libraries, but sacrifice fine-grained cross-token interactions.

In a Cross-Encoder architecture, query and document are concatenated as a single input sequence `[CLS] query [SEP] document [SEP]` and processed simultaneously through all transformer layers.
Every token in the query attends directly to every token in the document via full cross-attention.
While Cross-Encoders provide substantially superior ranking accuracy (nDCG@5), their $O(N)$ inference cost necessitates deploying them as a second-stage reranker over top candidate pools."""
    }
]


# Benchmark evaluation queries covering lexical, semantic, mixed, and unanswerable categories
BENCHMARK_QUERIES = [
    # Lexical Queries (exact technical terms)
    {
        "query_id": "q_lex_01",
        "query": "What is Python list comprehension and its canonical syntax?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_python_list_comprehension"],
        "graded_relevance": {"doc_python_list_comprehension": 3, "doc_python_generators_iterators": 1},
        "ground_truth_answer": "Python list comprehension creates new lists using the syntax [expression for item in iterable if condition]."
    },
    {
        "query_id": "q_lex_02",
        "query": "How does SQL INNER JOIN work when rows do not match?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dbms_sql_inner_join"],
        "graded_relevance": {"doc_dbms_sql_inner_join": 3},
        "ground_truth_answer": "Rows that do not match under the join predicate ON are omitted from the resulting relation."
    },
    {
        "query_id": "q_lex_03",
        "query": "What triggers a SQLSTATE 23505 error in PostgreSQL?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dbms_sqlstate_23505_errors"],
        "graded_relevance": {"doc_dbms_sqlstate_23505_errors": 3},
        "ground_truth_answer": "SQLSTATE 23505 indicates a unique constraint violation when inserting or updating duplicate keys."
    },
    {
        "query_id": "q_lex_04",
        "query": "What does an HTTP 404 error status code signify?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_systems_http_404_error"],
        "graded_relevance": {"doc_systems_http_404_error": 3, "doc_systems_http_status_codes_overview": 1},
        "ground_truth_answer": "HTTP 404 Not Found indicates that the web server cannot locate the requested resource at the specified URI."
    },
    {
        "query_id": "q_lex_05",
        "query": "What role does the gradient descent learning rate play?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_ml_gradient_descent_learning_rate"],
        "graded_relevance": {"doc_ml_gradient_descent_learning_rate": 3},
        "ground_truth_answer": "The learning rate controls parameter update step size; too small causes slow convergence, too large causes divergence."
    },
    {
        "query_id": "q_lex_06",
        "query": "What is the formula for TF-IDF inverse document frequency?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_nlp_tfidf_vectorization"],
        "graded_relevance": {"doc_nlp_tfidf_vectorization": 3, "doc_nlp_bm25_retrieval_ranking": 1},
        "ground_truth_answer": "IDF penalizes ubiquitous terms using the log ratio of total documents to document frequency."
    },
    {
        "query_id": "q_lex_07",
        "query": "What parameters tune Okapi BM25 retrieval?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_nlp_bm25_retrieval_ranking"],
        "graded_relevance": {"doc_nlp_bm25_retrieval_ranking": 3, "doc_nlp_tfidf_vectorization": 1},
        "ground_truth_answer": "Parameter k1 tunes term frequency saturation, and b tunes document length normalization."
    },
    {
        "query_id": "q_lex_08",
        "query": "What is the equation for Reciprocal Rank Fusion RRF?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_nlp_reciprocal_rank_fusion_rrf"],
        "graded_relevance": {"doc_nlp_reciprocal_rank_fusion_rrf": 3},
        "ground_truth_answer": "RRF sums 1 / (k0 + rank) across individual ranked retrieval lists."
    },
    {
        "query_id": "q_lex_09",
        "query": "How is Maximal Marginal Relevance MMR defined?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_nlp_maximal_marginal_relevance_mmr"],
        "graded_relevance": {"doc_nlp_maximal_marginal_relevance_mmr": 3},
        "ground_truth_answer": "MMR maximizes lambda * Sim(query, doc) - (1 - lambda) * max_sim(doc, selected)."
    },
    {
        "query_id": "q_lex_10",
        "query": "What is the formula for cross-entropy loss in classification?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dl_cross_entropy_loss_function"],
        "graded_relevance": {"doc_dl_cross_entropy_loss_function": 3},
        "ground_truth_answer": "Cross-entropy loss sums -y_c * log(y_hat_c) over all classes."
    },
    {
        "query_id": "q_lex_11",
        "query": "What is the difference between B-Tree and Hash index in databases?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dbms_indexing_btree_hash"],
        "graded_relevance": {"doc_dbms_indexing_btree_hash": 3},
        "ground_truth_answer": "B-Trees support range scans and equality in O(log N), whereas Hash indexes only support equality in O(1)."
    },
    {
        "query_id": "q_lex_12",
        "query": "What are the four ANSI SQL transaction isolation levels?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dbms_acid_transactions"],
        "graded_relevance": {"doc_dbms_acid_transactions": 3},
        "ground_truth_answer": "The four levels are Read Uncommitted, Read Committed, Repeatable Read, and Serializable."
    },
    {
        "query_id": "q_lex_13",
        "query": "What is the Global Interpreter Lock GIL in Python?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_python_gil_concurrency"],
        "graded_relevance": {"doc_python_gil_concurrency": 3},
        "ground_truth_answer": "The GIL is a mutex in CPython that prevents multiple native OS threads from executing bytecode simultaneously."
    },
    {
        "query_id": "q_lex_14",
        "query": "How does Python PyMalloc handle small memory allocations?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_python_memory_management"],
        "graded_relevance": {"doc_python_memory_management": 3},
        "ground_truth_answer": "PyMalloc manages a private heap optimized for small allocations up to 512 bytes."
    },
    {
        "query_id": "q_lex_15",
        "query": "What is the TCP three-way handshake packet sequence?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_systems_tcp_ip_three_way_handshake"],
        "graded_relevance": {"doc_systems_tcp_ip_three_way_handshake": 3},
        "ground_truth_answer": "The sequence consists of SYN from client, SYN-ACK from server, and ACK from client."
    },
    {
        "query_id": "q_lex_16",
        "query": "What is the formula for Z-score feature standardization?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_ml_feature_scaling_normalization"],
        "graded_relevance": {"doc_ml_feature_scaling_normalization": 3},
        "ground_truth_answer": "Z-score standardization transforms features to zero mean and unit variance via (x - mu) / sigma."
    },
    {
        "query_id": "q_lex_17",
        "query": "How does Batch Normalization normalize activations in deep networks?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dl_batch_normalization_layer"],
        "graded_relevance": {"doc_dl_batch_normalization_layer": 3},
        "ground_truth_answer": "BatchNorm computes mini-batch mean and variance, normalizes inputs, and applies learnable gamma and beta."
    },
    {
        "query_id": "q_lex_18",
        "query": "What is the Scaled Dot-Product Attention equation in transformers?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dl_transformer_attention_mechanism"],
        "graded_relevance": {"doc_dl_transformer_attention_mechanism": 3},
        "ground_truth_answer": "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."
    },
    {
        "query_id": "q_lex_19",
        "query": "What does HTTP status code 429 Too Many Requests mean?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_systems_http_status_codes_overview"],
        "graded_relevance": {"doc_systems_http_status_codes_overview": 3},
        "ground_truth_answer": "429 indicates client rate limiting and too many requests sent in a given time window."
    },
    {
        "query_id": "q_lex_20",
        "query": "What algorithm detects cycles in database wait-for graphs?",
        "query_type": "lexical",
        "relevant_doc_ids": ["doc_dbms_deadlock_detection"],
        "graded_relevance": {"doc_dbms_deadlock_detection": 3},
        "ground_truth_answer": "Engines run cycle detection on the transaction wait-for graph to abort a victim and break the deadlock."
    },

    # Semantic Queries (paraphrased, conceptual, no exact keyword match)
    {
        "query_id": "q_sem_01",
        "query": "How can I prevent a statistical model from memorizing the training dataset?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_ml_overfitting_regularization"],
        "graded_relevance": {"doc_ml_overfitting_regularization": 3, "doc_ml_bias_variance_tradeoff": 1},
        "ground_truth_answer": "Prevent memorization using L1/L2 regularization, early stopping, data augmentation, and dropout."
    },
    {
        "query_id": "q_sem_02",
        "query": "Why do uncollected references exhaust system RAM in Python applications?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_python_memory_management"],
        "graded_relevance": {"doc_python_memory_management": 3},
        "ground_truth_answer": "Cyclic references or lingering references in global dictionaries prevent reference counting from deallocating memory."
    },
    {
        "query_id": "q_sem_03",
        "query": "What architectural difference exists between joint attention ranking and separate vector comparison?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dl_cross_encoder_vs_bi_encoder"],
        "graded_relevance": {"doc_dl_cross_encoder_vs_bi_encoder": 3, "doc_nlp_dense_embeddings_semantic_search": 1},
        "ground_truth_answer": "Cross-encoders evaluate query and document simultaneously with cross-attention, whereas bi-encoders encode them independently."
    },
    {
        "query_id": "q_sem_04",
        "query": "How do database systems handle situations where two queries block each other indefinitely?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dbms_deadlock_detection"],
        "graded_relevance": {"doc_dbms_deadlock_detection": 3, "doc_dbms_acid_transactions": 1},
        "ground_truth_answer": "Engines detect cycles in a wait-for-graph and abort one transaction as the deadlock victim."
    },
    {
        "query_id": "q_sem_05",
        "query": "Why does Python restrict simultaneous multi-core execution in standard threads?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_python_gil_concurrency"],
        "graded_relevance": {"doc_python_gil_concurrency": 3},
        "ground_truth_answer": "The Global Interpreter Lock (GIL) prevents multiple native threads from executing bytecode simultaneously to protect memory management."
    },
    {
        "query_id": "q_sem_06",
        "query": "How can I avoid duplicate side-effects when an API request is retried multiple times?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_systems_rest_api_idempotency"],
        "graded_relevance": {"doc_systems_rest_api_idempotency": 3},
        "ground_truth_answer": "Use idempotent HTTP methods or attach unique Idempotency-Key headers verified in a cache."
    },
    {
        "query_id": "q_sem_07",
        "query": "What mechanism allows functions to remember outer lexical variables after return?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_python_decorators_closures"],
        "graded_relevance": {"doc_python_decorators_closures": 3},
        "ground_truth_answer": "Lexical closures retain references to outer scope variables within nested function environments."
    },
    {
        "query_id": "q_sem_08",
        "query": "How does scaling the dot product prevent vanishing gradients in transformers?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dl_transformer_attention_mechanism"],
        "graded_relevance": {"doc_dl_transformer_attention_mechanism": 3},
        "ground_truth_answer": "Dividing by sqrt(d_k) prevents large dot products from pushing softmax into regions with near-zero gradients."
    },
    {
        "query_id": "q_sem_09",
        "query": "Why do tree-structured indexes handle ordering and ranges better than hash structures?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dbms_indexing_btree_hash"],
        "graded_relevance": {"doc_dbms_indexing_btree_hash": 3},
        "ground_truth_answer": "B-Tree leaf nodes are maintained in sorted order and linked sequentially, unlike hash tables."
    },
    {
        "query_id": "q_sem_10",
        "query": "How do models normalize internal layer activations across mini-batches during training?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dl_batch_normalization_layer"],
        "graded_relevance": {"doc_dl_batch_normalization_layer": 3},
        "ground_truth_answer": "Batch Normalization computes mini-batch mean and variance, normalizes activations, and scales with learnable gamma and beta."
    },
    {
        "query_id": "q_sem_11",
        "query": "How does computing derivatives in reverse topological order save computational complexity?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dl_backpropagation_chain_rule"],
        "graded_relevance": {"doc_dl_backpropagation_chain_rule": 3},
        "ground_truth_answer": "Reverse-mode autodiff calculates sensitivities from loss to inputs in time proportional to a single forward pass."
    },
    {
        "query_id": "q_sem_12",
        "query": "Why does averaging multiple diverse predictors reduce prediction variability without hurting accuracy?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_ml_bias_variance_tradeoff"],
        "graded_relevance": {"doc_ml_bias_variance_tradeoff": 3},
        "ground_truth_answer": "Bagging averages de-correlated estimators, reducing overall variance while preserving low individual bias."
    },
    {
        "query_id": "q_sem_13",
        "query": "What ensures database readers and writers never block each other concurrently?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dbms_acid_transactions"],
        "graded_relevance": {"doc_dbms_acid_transactions": 3},
        "ground_truth_answer": "Multi-Version Concurrency Control (MVCC) maintains snapshot versions of tuples so readers do not lock writers."
    },
    {
        "query_id": "q_sem_14",
        "query": "How does domain name lookup discover canonical machine addresses step by step?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_systems_dns_resolution_lifecycle"],
        "graded_relevance": {"doc_systems_dns_resolution_lifecycle": 3},
        "ground_truth_answer": "Resolution traverses local cache, recursive resolvers, root nameservers, TLD servers, and authoritative nameservers."
    },
    {
        "query_id": "q_sem_15",
        "query": "Why should feature scaling parameters strictly avoid touching evaluation splits during preprocessing?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_ml_cross_validation_strategies"],
        "graded_relevance": {"doc_ml_cross_validation_strategies": 3},
        "ground_truth_answer": "Fitting scalers on validation data constitutes data leakage, artificially inflating validation performance."
    },
    {
        "query_id": "q_sem_16",
        "query": "How do generator functions suspend execution and resume later without losing state?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_python_generators_iterators"],
        "graded_relevance": {"doc_python_generators_iterators": 3},
        "ground_truth_answer": "The yield keyword pauses the generator frame, preserving local variables and instruction pointer."
    },
    {
        "query_id": "q_sem_17",
        "query": "What technique prevents near-duplicate text chunks from crowding the LLM prompt window?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_nlp_maximal_marginal_relevance_mmr"],
        "graded_relevance": {"doc_nlp_maximal_marginal_relevance_mmr": 3},
        "ground_truth_answer": "Maximal Marginal Relevance penalizes candidates highly similar to already selected chunks."
    },
    {
        "query_id": "q_sem_18",
        "query": "How do hash join algorithms merge two relational tables in memory efficiently?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_dbms_sql_inner_join"],
        "graded_relevance": {"doc_dbms_sql_inner_join": 3},
        "ground_truth_answer": "Hash joins construct an in-memory hash table on the smaller relation and probe it with the larger relation in O(M+N)."
    },
    {
        "query_id": "q_sem_19",
        "query": "Why do outliers damage Min-Max scaling more severely than Z-score standardization?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_ml_feature_scaling_normalization"],
        "graded_relevance": {"doc_ml_feature_scaling_normalization": 3},
        "ground_truth_answer": "Min-Max scaling uses extreme min and max values in the denominator, squashing all typical inliers into a narrow band."
    },
    {
        "query_id": "q_sem_20",
        "query": "How does the Python cyclical garbage collector identify unreferenced cycles?",
        "query_type": "semantic",
        "relevant_doc_ids": ["doc_python_memory_management"],
        "graded_relevance": {"doc_python_memory_management": 3},
        "ground_truth_answer": "The generational GC tracks container objects across three generations and isolates unreachable reference cycles."
    },

    # Mixed Queries (conversational phrasing + specific technical entities)
    {
        "query_id": "q_mix_01",
        "query": "Could you please explain how to combine BM25 and dense retrieval with RRF fusion?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_nlp_reciprocal_rank_fusion_rrf"],
        "graded_relevance": {"doc_nlp_reciprocal_rank_fusion_rrf": 3, "doc_nlp_bm25_retrieval_ranking": 2, "doc_nlp_dense_embeddings_semantic_search": 2},
        "ground_truth_answer": "Retrieve top candidates from BM25 and dense search, then fuse ranks using the formula 1 / (60 + rank)."
    },
    {
        "query_id": "q_mix_02",
        "query": "I want to know what causes an HTTP 404 client error in single page applications.",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_systems_http_404_error"],
        "graded_relevance": {"doc_systems_http_404_error": 3, "doc_systems_http_status_codes_overview": 1},
        "ground_truth_answer": "In SPAs, servers must rewrite unmatched paths to index.html to allow client-side routing instead of returning 404."
    },
    {
        "query_id": "q_mix_03",
        "query": "Can you tell me how to prevent duplicate key collisions under SQLSTATE 23505?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_dbms_sqlstate_23505_errors"],
        "graded_relevance": {"doc_dbms_sqlstate_23505_errors": 3},
        "ground_truth_answer": "Use INSERT ... ON CONFLICT DO UPDATE or DO NOTHING upsert statements."
    },
    {
        "query_id": "q_mix_04",
        "query": "Tell me about tuning the gradient descent learning rate with cosine annealing warmup.",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_ml_gradient_descent_learning_rate"],
        "graded_relevance": {"doc_ml_gradient_descent_learning_rate": 3},
        "ground_truth_answer": "Decaying the learning rate with schedules like cosine annealing with warmup stabilizes late training convergence."
    },
    {
        "query_id": "q_mix_05",
        "query": "Please describe the difference between Python list comprehensions and generator memory usage.",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_python_list_comprehension"],
        "graded_relevance": {"doc_python_list_comprehension": 3, "doc_python_generators_iterators": 1},
        "ground_truth_answer": "List comprehensions allocate full lists eagerly in memory, whereas generator expressions yield items lazily with O(1) memory."
    },
    {
        "query_id": "q_mix_06",
        "query": "Could you explain how cross-encoders differ from bi-encoders in ranking accuracy and latency?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_dl_cross_encoder_vs_bi_encoder"],
        "graded_relevance": {"doc_dl_cross_encoder_vs_bi_encoder": 3},
        "ground_truth_answer": "Cross-encoders achieve higher accuracy via full cross-attention but are slower, making them suitable as 2nd stage rerankers."
    },
    {
        "query_id": "q_mix_07",
        "query": "What are the rules for choosing between L1 lasso and L2 ridge regularization to avoid overfitting?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_ml_overfitting_regularization"],
        "graded_relevance": {"doc_ml_overfitting_regularization": 3},
        "ground_truth_answer": "L1 induces parameter sparsity for feature selection, while L2 shrinks weights smoothly to prevent extreme magnitudes."
    },
    {
        "query_id": "q_mix_08",
        "query": "How can I implement an idempotent payment API endpoint using HTTP POST and Redis?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_systems_rest_api_idempotency"],
        "graded_relevance": {"doc_systems_rest_api_idempotency": 3},
        "ground_truth_answer": "Send unique Idempotency-Key headers verified and locked in Redis to prevent duplicate processing."
    },
    {
        "query_id": "q_mix_09",
        "query": "What causes TCP connections to enter the TIME_WAIT state after teardown?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_systems_tcp_ip_three_way_handshake"],
        "graded_relevance": {"doc_systems_tcp_ip_three_way_handshake": 3},
        "ground_truth_answer": "The client waits in TIME_WAIT to ensure final ACKs are received and lingering duplicate segments expire safely."
    },
    {
        "query_id": "q_mix_10",
        "query": "Please explain how multi-head attention computes representation subspaces in transformers.",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_dl_transformer_attention_mechanism"],
        "graded_relevance": {"doc_dl_transformer_attention_mechanism": 3},
        "ground_truth_answer": "Multi-head attention projects queries, keys, and values into h distinct linear subspaces."
    },
    {
        "query_id": "q_mix_11",
        "query": "How do database engines abort deadlocks using wait-for-graph victim selection?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_dbms_deadlock_detection"],
        "graded_relevance": {"doc_dbms_deadlock_detection": 3},
        "ground_truth_answer": "When a directed cycle is detected in the lock wait-for graph, one transaction is selected as victim and rolled back."
    },
    {
        "query_id": "q_mix_12",
        "query": "What is the relationship between TF-IDF inverse document frequency and BM25 term weighting?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_nlp_bm25_retrieval_ranking"],
        "graded_relevance": {"doc_nlp_bm25_retrieval_ranking": 3, "doc_nlp_tfidf_vectorization": 2},
        "ground_truth_answer": "Both weight terms inversely by document frequency, but BM25 applies asymptotic term frequency saturation."
    },
    {
        "query_id": "q_mix_13",
        "query": "How can I avoid GIL bottlenecks for CPU-bound tasks in Python scripts?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_python_gil_concurrency"],
        "graded_relevance": {"doc_python_gil_concurrency": 3},
        "ground_truth_answer": "Use multiprocessing or ProcessPoolExecutor to run independent Python processes with their own GILs."
    },
    {
        "query_id": "q_mix_14",
        "query": "What happens when an HTTP request receives a 502 Bad Gateway response?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_systems_http_status_codes_overview"],
        "graded_relevance": {"doc_systems_http_status_codes_overview": 3},
        "ground_truth_answer": "A 502 Bad Gateway indicates that a reverse proxy or edge gateway received an invalid response from upstream servers."
    },
    {
        "query_id": "q_mix_15",
        "query": "How does cross-entropy loss simplify when paired with a softmax output activation?",
        "query_type": "mixed",
        "relevant_doc_ids": ["doc_dl_cross_entropy_loss_function"],
        "graded_relevance": {"doc_dl_cross_entropy_loss_function": 3},
        "ground_truth_answer": "The derivative with respect to each logit simplifies directly to predicted probability minus ground truth indicator."
    },

    # Unanswerable Queries (to test anti-hallucination and rejection)
    {
        "query_id": "q_unans_01",
        "query": "What is the capital city of France and its total population?",
        "query_type": "unanswerable",
        "relevant_doc_ids": [],
        "graded_relevance": {},
        "ground_truth_answer": "I don't have enough information in the provided documents to answer this question."
    },
    {
        "query_id": "q_unans_02",
        "query": "How do you configure an Apache Cassandra gossip protocol cluster?",
        "query_type": "unanswerable",
        "relevant_doc_ids": [],
        "graded_relevance": {},
        "ground_truth_answer": "I don't have enough information in the provided documents to answer this question."
    },
    {
        "query_id": "q_unans_03",
        "query": "What are the rules of chess regarding en passant capture?",
        "query_type": "unanswerable",
        "relevant_doc_ids": [],
        "graded_relevance": {},
        "ground_truth_answer": "I don't have enough information in the provided documents to answer this question."
    },
    {
        "query_id": "q_unans_04",
        "query": "Who won the FIFA World Cup in 1994?",
        "query_type": "unanswerable",
        "relevant_doc_ids": [],
        "graded_relevance": {},
        "ground_truth_answer": "I don't have enough information in the provided documents to answer this question."
    },
    {
        "query_id": "q_unans_05",
        "query": "What is the chemical boiling point of liquid nitrogen at 1 atmosphere?",
        "query_type": "unanswerable",
        "relevant_doc_ids": [],
        "graded_relevance": {},
        "ground_truth_answer": "I don't have enough information in the provided documents to answer this question."
    }
]


def generate_all_data(base_dir: Path) -> None:
    """Writes all 30 technical documents and evaluation dataset files to disk."""
    raw_dir = base_dir / "data" / "raw"
    eval_dir = base_dir / "data" / "evaluation"
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write documents
    for doc in DOCUMENTS:
        file_path = raw_dir / doc["filename"]
        file_path.write_text(doc["content"].strip(), encoding="utf-8")

    # 2. Write benchmark queries
    queries_file = eval_dir / "benchmark_queries.json"
    queries_file.write_text(json.dumps(BENCHMARK_QUERIES, indent=2), encoding="utf-8")
    print(f"Generated {len(DOCUMENTS)} documents in {raw_dir}")
    print(f"Generated {len(BENCHMARK_QUERIES)} benchmark queries in {queries_file}")


if __name__ == "__main__":
    generate_all_data(Path(__file__).resolve().parent.parent)
