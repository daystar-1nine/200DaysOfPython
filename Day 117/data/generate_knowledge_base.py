"""
Synthesizes the multi-domain reference knowledge base and ground-truth evaluation dataset.
Creates 30 focused technical documents across 7 domains yielding ~150 chunks at chunk_size=400.
"""
from pathlib import Path
import json

base_dir = Path(__file__).resolve().parent
raw_dir = base_dir / "raw"
eval_dir = base_dir / "evaluation"

DOCUMENTS = {
    # 1. Python
    "python/python_data_structures.md": """# Python Built-in Data Structures and Mutability

Python provides several core built-in data structures optimized for distinct access patterns. Lists are ordered, mutable sequences defined with square brackets, supporting dynamic resizing, amortized O(1) append and pop from the end, and in-place sorting. In contrast, tuples are ordered, immutable sequences defined with parentheses; their immutability allows them to be hashed and used as dictionary keys when their constituent elements are also immutable.

Sets are unordered collections of unique, hashable elements implemented via hash tables, providing O(1) average-time complexity for lookups, insertions, and membership tests via in keyword. Dictionaries represent associative arrays mapping unique keys to arbitrary values, preserving insertion order since Python 3.7. Under the hood, Python dictionaries use a compact array combined with a sparse hash table to minimize memory overhead while retaining O(1) key lookups.
""",

    "python/python_generators_comprehensions.md": """# Python Comprehensions, Generators, and Lazy Evaluation

List comprehensions provide a concise syntax for mapping and filtering iterables, constructing a full in-memory list upon execution: [x*2 for x in items if x > 0]. While highly expressive and fast due to optimized C-level bytecode loops, list comprehensions can consume excessive RAM when dealing with large or infinite sequences.

Generator expressions solve this memory bottleneck by using parentheses instead of square brackets: (x*2 for x in items). Generators produce values lazily on-demand using the iterator protocol (__iter__ and __next__). Generator functions use the yield keyword to pause execution and yield intermediate values, maintaining internal state across invocations. The itertools module further provides memory-efficient combinatoric iterators such as chain, islice, and cycle.
""",

    "python/python_decorators_functions.md": """# Python Decorators, Closures, and First-Class Functions

In Python, functions are first-class citizens, meaning they can be passed as arguments, returned from other functions, assigned to variables, and stored in collections. A closure is a function that retains access to variables in its enclosing lexical scope even after that outer scope has finished execution.

A decorator is a callable that accepts another function as an argument, wraps or extends its behavior without modifying the underlying source code, and returns the modified wrapper function. Decorators commonly implement cross-cutting concerns such as logging, execution timing, input validation, authentication, and caching. Using functools.wraps preserves the decorated function's original docstring, name, and signature introspection.
""",

    "python/python_concurrency_gil.md": """# Global Interpreter Lock (GIL) and Concurrency in Python

The CPython implementation relies on the Global Interpreter Lock (GIL), a mutual-exclusion lock preventing multiple native operating system threads from executing Python bytecode concurrently. While the GIL protects CPython memory management and reference counting from race conditions, it limits CPU-bound multi-threading to a single physical core.

To achieve true parallelism for CPU-bound workloads, Python programs use multiprocessing, which spawns separate operating system processes with independent memory spaces and separate GIL instances. Conversely, I/O-bound operations (such as network requests or disk reads) release the GIL while waiting for external resources, making threading or cooperative multitasking with asyncio highly effective for concurrent I/O throughput.
""",

    "python/python_dunder_oop.md": """# Object-Oriented Programming and Dunder Methods in Python

Python object-oriented programming relies on special double-underscore ("dunder") methods to customize class behaviors and integrate with built-in language semantics. The __init__ method initializes newly instantiated objects, __repr__ provides an unambiguous developer representation, and __str__ yields an informal human-readable string representation.

Operator overloading is accomplished through arithmetic dunders like __add__ and comparison dunders like __eq__ and __lt__. Container emulation uses __len__, __getitem__, and __setitem__. Context managers implement __enter__ and __exit__ to ensure deterministic resource allocation and reclamation (such as closing database connections or file handles) via with statements.
""",

    # 2. Data Science
    "data_science/numpy_vectorization.md": """# NumPy Arrays, Vectorization, and SIMD Operations

NumPy forms the fundamental numerical substrate of Python data science. The core ndarray object stores homogeneous, contiguous memory buffers, enabling vectorized arithmetic operations implemented in optimized C, Fortran, and BLAS/LAPACK routines. Vectorization replaces explicit Python for-loops with SIMD (Single Instruction, Multiple Data) processor instructions, achieving orders-of-magnitude computational speedups.

NumPy broadcasting allows arithmetic operations between arrays of differing shapes by aligning dimensions from right to left. Two dimensions are compatible if they are equal or if one of them is 1. Singleton dimensions are conceptually repeated to match the target shape without copying underlying memory buffers, preserving cache efficiency.
""",

    "data_science/pandas_indexing_selection.md": """# Pandas Indexing, Selection, and Memory Semantics

Pandas structures tabular data using DataFrames (2D tables) and Series (1D labelled arrays). Data selection distinguishes between label-based indexing with .loc[] and integer position-based indexing with .iloc[]. Using boolean masks allows intuitive row filtering based on complex conditional logic.

Chained assignment (such as df[mask]['col'] = val) triggers SettingWithCopyWarning because it is ambiguous whether a view or an in-memory copy was returned by the intermediate slice. Explicit assignment using .loc[mask, 'col'] = val guarantees reliable in-place mutation and avoids silent data corruption.
""",

    "data_science/pandas_groupby_aggregations.md": """# Pandas Split-Apply-Combine and GroupBy Operations

The Split-Apply-Combine paradigm divides tabular datasets into discrete groups based on specified grouping keys, applies transformation or reduction functions to each group independently, and combines the results into a unified structure. The .groupby() method supports aggregate functions (mean, sum, count, std), transformations that retain the original DataFrame index, and filtering operations that discard entire groups failing specific predicate criteria.

Rolling window calculations allow computing moving averages, rolling standard deviations, and cumulative sums across ordered temporal data. Combining groupby with rolling or expanding windows enables powerful localized statistical feature engineering for time series forecasting.
""",

    "data_science/data_cleaning_missing_outliers.md": """# Handling Missing Data and Outliers in Tabular Datasets

Real-world datasets frequently contain missing values represented as NaN or None. Pandas provides .isna(), .fillna(), and .dropna() for missing value imputation and deletion. Imputation strategies range from simple mean, median, or mode replacement to forward/backward filling for time series, and iterative algorithms like MICE (Multivariate Imputation by Chained Equations).

Outliers are typically detected using parametric Z-scores (values beyond 3 standard deviations from the mean) or non-parametric Interquartile Range (IQR) fences (values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR). Capping outliers via Winsorization or transforming skewed distributions via log or Box-Cox transformations prevents extreme values from distorting linear models.
""",

    # 3. Machine Learning
    "machine_learning/bias_variance_tradeoff.md": """# The Bias-Variance Tradeoff in Supervised Learning

The bias-variance tradeoff governs generalization error in supervised learning. Bias measures the error originating from erroneous assumptions or over-simplifications in the model learning algorithm, leading to underfitting where the model fails to capture underlying data patterns. Variance measures the model's sensitivity to small fluctuations in the training dataset, leading to overfitting where the model memorizes noise rather than signal.

The total expected generalization error decomposes into Bias^2 + Variance + Irreducible Error. Highly flexible models (such as deep decision trees or unregularized neural networks) typically exhibit low bias but high variance, whereas rigid models (such as linear regression) exhibit high bias and low variance. Cross-validation and regularization help locate the optimal tradeoff point.
""",

    "machine_learning/gradient_descent_optimization.md": """# Gradient Descent Optimization Algorithms in Machine Learning

Gradient descent minimizes empirical risk loss functions by iteratively updating parameter weights in the direction opposite to the gradient of the loss with respect to parameters: w = w - lr * grad. Batch Gradient Descent computes the exact gradient over the entire training corpus, ensuring smooth monotonic convergence but incurring heavy computational and memory costs for large datasets.

Stochastic Gradient Descent (SGD) updates parameters using a single random sample per iteration, introducing high stochastic variance that aids in escaping shallow local minima but leads to noisy convergence paths. Mini-batch Gradient Descent balances both paradigms by computing gradients over small subsets (e.g. 32 to 256 samples), maximizing hardware GPU parallelism while providing robust gradient estimates.
""",

    "machine_learning/regularization_l1_l2.md": """# Regularization Techniques: L1 Lasso and L2 Ridge

Regularization prevents machine learning models from overfitting by penalizing large parameter weights in the optimization objective. L1 regularization (Lasso) adds the sum of absolute parameter weights to the loss function (lambda * sum(|w|)). Because the L1 penalty has non-differentiable sharp corners at zero, it drives non-essential coefficients exactly to zero, effectively performing embedded feature selection and producing sparse models.

L2 regularization (Ridge) adds the sum of squared parameter weights (lambda * sum(w^2)), shrinking coefficients continuously toward zero without eliminating them completely. Elastic Net combines both L1 and L2 penalties with a mixing ratio parameter, successfully handling correlated feature groups where Lasso might arbitrarily select only one predictor.
""",

    "machine_learning/ensemble_tree_methods.md": """# Ensemble Learning: Bagging, Random Forests, and Gradient Boosting

Ensemble methods combine multiple base estimators to produce superior predictive accuracy and robustness. Bagging (Bootstrap Aggregating) trains diverse base models in parallel on independent bootstrap samples drawn with replacement from the training set, reducing variance without increasing bias. Random Forests extend bagging by additionally selecting random feature subsets at each tree split, de-correlating individual decision trees.

Boosting trains weak learners sequentially, where each successive estimator focuses on correcting the residual errors or misclassifications of preceding models. Modern gradient boosting frameworks like XGBoost, LightGBM, and CatBoost implement regularized objective functions, second-order Taylor approximations of the loss function, and histogram-based split finding for exceptional efficiency on tabular data.
""",

    # 4. Deep Learning
    "deep_learning/neural_networks_backpropagation.md": """# Artificial Neural Networks and the Backpropagation Algorithm

Artificial neural networks compose layers of interconnected artificial neurons computing affine matrix transformations followed by non-linear activation functions: y = f(W*x + b). Multi-layer perceptrons (MLPs) act as universal function approximators capable of modeling non-linear decision boundaries.

The backpropagation algorithm calculates the partial derivatives of the scalar loss with respect to every weight and bias parameter in the network via the chain rule of calculus. Error signals are propagated backwards from the output loss layer through hidden layers to input layers. Modern deep learning frameworks construct dynamic computation graphs during the forward pass to automatically compute and accumulate these gradients during the backward pass.
""",

    "deep_learning/activations_vanishing_gradients.md": """# Vanishing Gradients and Non-linear Activation Functions

During backpropagation through deep neural networks, gradients can shrink exponentially if activation functions saturate. Classical sigmoid and tanh activations compress inputs into bounded ranges where derivative magnitudes peak below 0.25 and 1.0 respectively; multiplying these fractional derivatives across many layers drives early gradients toward zero, halting learning.

The Rectified Linear Unit (ReLU) activation, defined as max(0, x), avoids vanishing gradients for positive inputs because its derivative is exactly 1.0. However, ReLU can suffer from the 'dying ReLU' phenomenon when neurons receive large negative updates and permanently output zero. Variants like LeakyReLU, Parametric ReLU (PReLU), and Gaussian Error Linear Units (GELU) provide smooth non-zero gradients for negative inputs.
""",

    "deep_learning/optimizers_batchnorm.md": """# Adaptive Optimization Algorithms and Batch Normalization

Modern deep neural networks rely on adaptive learning rate optimizers. The Adam (Adaptive Moment Estimation) optimizer tracks exponential moving averages of both past gradients (first moment, momentum) and squared gradients (second moment, uncentered variance), providing adaptive per-parameter learning rate scaling and robust convergence across sparse gradients.

Batch Normalization (BatchNorm) normalizes activations across the mini-batch dimension during training: x_hat = (x - mean) / sqrt(var + eps), followed by learnable scale (gamma) and shift (beta) transformations. BatchNorm stabilizes internal covariate shift, smooths the optimization loss landscape, and allows significantly higher learning rates while acting as a mild regularizer.
""",

    "deep_learning/transformers_self_attention.md": """# Transformer Self-Attention and Scaled Dot-Product Mechanics

The Transformer architecture eliminates recurrence, processing entire sequences concurrently using scaled dot-product attention: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V. Query (Q), Key (K), and Value (V) matrices are linearly projected from input token representations. Dividing the dot product by sqrt(d_k) prevents large magnitudes in high-dimensional spaces from pushing the softmax function into regions with vanishingly small gradients.

Multi-head attention splits projections into multiple representation subspaces, enabling the model to attend simultaneously to syntactic, semantic, and positional dependencies across tokens. Positional encodings (sinusoidal or learned rotary RoPE embeddings) inject sequence order information into the permutation-invariant attention mechanism.
""",

    # 5. Computer Vision
    "computer_vision/convolutional_neural_networks.md": """# Convolutional Neural Networks (CNNs) and Spatial Feature Hierarchies

Convolutional Neural Networks process visual grid data by applying localized 2D filter kernels across spatial dimensions. Convolutions preserve 2D spatial topology and leverage weight sharing, drastically reducing parameter counts compared to fully connected networks. Deeper layers capture increasingly abstract hierarchical features, progressing from low-level edges and textures to complex object parts.

Pooling layers (such as Max Pooling and Average Pooling) downsample spatial feature maps, conferring translation invariance and expanding the effective receptive field of subsequent convolutional layers. Strided convolutions can also perform spatial downsampling while learning parameterized decimation kernels.
""",

    "computer_vision/residual_networks.md": """# Residual Networks (ResNets) and Identity Skip Connections

Prior to Residual Networks (ResNets), stacking dozens of convolutional layers caused degradation in training accuracy due to optimization difficulties rather than overfitting. ResNets resolve this degradation by introducing identity skip connections (shortcut connections) where input x bypasses weight layers: F(x) + x.

These residual paths allow gradients to flow directly backwards through the network without attenuation, enabling stable gradient backpropagation in architectures exceeding 100 or 1000 layers. Bottleneck residual blocks use 1x1 convolutions before and after 3x3 convolutions to reduce and restore feature dimensions, minimizing computational cost while maximizing representational capacity.
""",

    "computer_vision/object_detection_architectures.md": """# Object Detection: Two-Stage vs One-Stage Detectors

Object detection identifies spatial bounding boxes and semantic class labels for multiple objects in an image. Two-stage detectors (such as Faster R-CNN) first generate sparse candidate regions of interest using a Region Proposal Network (RPN) and subsequently classify and refine bounding box coordinates using RoI pooling or RoIAlign.

One-stage detectors (such as YOLO - You Only Look Once, and SSD) frame detection as a single unified regression problem, predicting bounding boxes and class probabilities directly from dense feature map grids in a single forward pass. While two-stage detectors traditionally achieved higher localization accuracy, modern one-stage detectors achieve comparable accuracy at real-time inference speeds (exceeding 60 FPS).
""",

    "computer_vision/detection_metrics_nms.md": """# Detection Evaluation Metrics: IoU, NMS, and mAP

Intersection over Union (IoU) measures spatial overlap by dividing the area of overlap by the area of union between two bounding boxes: IoU = Area(A cap B) / Area(A cup B). An IoU threshold of 0.5 or 0.7 determines whether a predicted bounding box is considered a True Positive.

Non-Maximum Suppression (NMS) eliminates duplicate candidate boxes for the same physical object. NMS greedily selects the highest-scoring candidate box and discards all neighboring boxes whose IoU overlap exceeds a predefined threshold. Mean Average Precision (mAP) evaluates overall detector performance by averaging precision across recall levels and semantic categories (e.g. mAP@0.5 and mAP@0.5:0.95).
""",

    # 6. Natural Language Processing
    "nlp/tokenization_subword_algorithms.md": """# Subword Tokenization Algorithms in Modern NLP

Tokenization converts raw character strings into numerical token IDs suitable for neural models. Classical word-level tokenization struggles with out-of-vocabulary (OOV) terms and morphological variations, while character-level tokenization produces excessively long sequence lengths that stress attention mechanisms.

Modern NLP relies on subword tokenization algorithms. Byte-Pair Encoding (BPE) starts with individual characters and iteratively merges the most frequent adjacent character pairs into new tokens. WordPiece selects merges that maximize language model likelihood. Unigram Tokenizer starts with a large candidate vocabulary and iteratively prunes tokens that cause the smallest increase in unigram loss. Byte-level BPE (used in GPT) operates directly on UTF-8 bytes to handle any Unicode string without unknown token artifacts.
""",

    "nlp/statistical_nlp_tfidf.md": """# Statistical Text Representation: TF-IDF and Bag-of-Words

The Bag-of-Words model represents text documents as unordered frequency histograms of constituent vocabulary tokens, discarding grammatical syntax and word order. Term Frequency-Inverse Document Frequency (TF-IDF) refines Bag-of-Words by weighting tokens according to their corpus-wide specificity.

Term Frequency (TF) measures how frequently a term appears in a given document: TF(t, d) = count(t, d) / total_words(d). Inverse Document Frequency (IDF) penalizes ubiquitous words (such as 'the' or 'is') that appear across almost all documents: IDF(t) = log(N / (df(t) + 1)). The composite weight TF*IDF identifies distinctive domain keywords, creating high-dimensional sparse document vectors suitable for lexical search and text classification.
""",

    "nlp/dense_embeddings_word2vec.md": """# Dense Word Embeddings: Word2Vec, GloVe, and Semantic Geometry

Dense word embeddings represent vocabulary tokens as continuous, low-dimensional vectors (e.g. 100 to 768 dimensions) where semantically related words occupy proximate coordinates in geometric vector space. Word2Vec learns these representations using shallow neural architectures: Continuous Bag of Words (CBOW) predicts target center words from context windows, while Skip-gram predicts surrounding context words given a center word using negative sampling.

Dense embeddings exhibit linear semantic compositionality, famously demonstrated by vector arithmetic: vector('King') - vector('Man') + vector('Woman') ≈ vector('Queen'). Global Vectors (GloVe) achieves comparable representations by factorizing global log-bilinear word co-occurrence matrices across the entire corpus.
""",

    "nlp/retrieval_augmented_generation.md": """# Retrieval-Augmented Generation (RAG) Architecture and Grounding

Retrieval-Augmented Generation (RAG) combines neural information retrieval with autoregressive language generation. Rather than relying solely on parametric knowledge memorized in neural network weights during pretraining, a RAG system dynamically retrieves non-parametric factual chunks from an external indexed document store based on the user query.

The retrieved chunks are concatenated into a structured grounding prompt supplied to the language model. This decoupled design mitigates hallucinations, circumvents training knowledge cutoff limitations, protects private internal organizational data, and enables verifiable source attribution through explicit citation tags.
""",

    # 7. Database Management Systems
    "dbms/acid_transactions_wal.md": """# ACID Properties and Write-Ahead Logging in Relational Databases

Relational database management systems (RDBMS) guarantee transactional reliability through ACID properties. Atomicity ensures that all operations in a transaction execute successfully or the entire transaction aborts with zero side effects. Consistency guarantees that transactions transition the database from one valid state to another, preserving integrity constraints.

Isolation ensures concurrent transactions execute without interference, governed by isolation levels: Read Uncommitted, Read Committed, Repeatable Read, and Serializable. Durability guarantees that committed modifications persist permanently even across catastrophic system crashes. Durability is implemented via Write-Ahead Logging (WAL), where log records describing modifications are flushed to non-volatile disk before dirty memory pages are written to the database data files.
""",

    "dbms/relational_normalization.md": """# Relational Database Normalization: 1NF, 2NF, 3NF, and BCNF

Normalization organizes database relational schemas to minimize data redundancy and prevent insertion, update, and deletion anomalies. First Normal Form (1NF) requires atomic attribute values and unique primary keys. Second Normal Form (2NF) builds upon 1NF by eliminating partial dependencies, ensuring that all non-key attributes depend on the complete composite primary key.

Third Normal Form (3NF) eliminates transitive dependencies, requiring that non-key attributes depend only on candidate keys (no non-key attribute determines another non-key attribute). Boyce-Codd Normal Form (BCNF) strictly requires that for every functional dependency X -> Y, X must be a superkey. Higher normal forms reduce anomalies at the expense of requiring additional SQL joins during retrieval.
""",

    "dbms/database_indexes_btrees.md": """# Database Indexing: B-Trees, B+Trees, and Hash Indexes

Database indexes accelerate record retrieval queries at the cost of additional write latency and storage overhead. The B+Tree index is the standard indexing structure for relational engines. It maintains a balanced, multi-way search tree where internal nodes store guiding routing keys and leaf nodes store actual record pointers or table row data.

B+Trees store all data pointers exclusively in leaf nodes, which are linked together in a bidirectional doubly-linked list. This architecture makes B+Trees exceptionally efficient for range scans, prefix searches, and ordered sorting (ORDER BY) in O(log N) time. In contrast, Hash Indexes provide O(1) point lookups based on bucket hashing but cannot support range queries or sorting.
""",

    "dbms/sql_joins_query_optimization.md": """# SQL Join Algorithms and Cost-Based Query Optimization

Relational database engines combine records across tables using join operations. Inner Joins return matching rows from both relations; Left Outer Joins return all rows from the left relation with matching rows from the right relation (filling unmatched attributes with NULLs).

The Cost-Based Query Optimizer evaluates alternative physical execution plans using catalog statistics (histograms, table row counts, index selectivity). It chooses between three fundamental join algorithms: Nested Loop Join (optimal when joining small subsets against an indexed table), Hash Join (optimal for large unindexed equijoins by building an in-memory hash table on the smaller relation), and Sort-Merge Join (optimal when both input relations are already sorted on the join key).
"""
}

# 60 Ground-Truth Benchmark Questions
EVAL_QUESTIONS = [
    # --- Python (8 questions) ---
    {
        "id": "q001",
        "category": "python",
        "question": "What is the difference between a list and a tuple in Python?",
        "relevant_docs": ["doc_python_data_structures"],
        "relevant_chunks": ["doc_python_data_structures_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q002",
        "category": "python",
        "question": "Why do generator expressions use less memory than list comprehensions?",
        "relevant_docs": ["doc_python_generators_comprehensions"],
        "relevant_chunks": ["doc_python_generators_comprehensions_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q003",
        "category": "python",
        "question": "Explain what a Python decorator is and how it works.",
        "relevant_docs": ["doc_python_decorators_functions"],
        "relevant_chunks": ["doc_python_decorators_functions_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q004",
        "category": "python",
        "question": "What is the Global Interpreter Lock (GIL) in CPython?",
        "relevant_docs": ["doc_python_concurrency_gil"],
        "relevant_chunks": ["doc_python_concurrency_gil_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q005",
        "category": "python",
        "question": "Which dunder methods implement context managers for with statements?",
        "relevant_docs": ["doc_python_dunder_oop"],
        "relevant_chunks": ["doc_python_dunder_oop_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q006",
        "category": "python",
        "question": "How do sets achieve O(1) average lookup time in Python?",
        "relevant_docs": ["doc_python_data_structures"],
        "relevant_chunks": ["doc_python_data_structures_c0"],
        "supported": True,
        "query_type": "paraphrased"
    },
    {
        "id": "q007",
        "category": "python",
        "question": "GIL Python",
        "relevant_docs": ["doc_python_concurrency_gil"],
        "relevant_chunks": ["doc_python_concurrency_gil_c0"],
        "supported": True,
        "query_type": "short"
    },
    {
        "id": "q008",
        "category": "python",
        "question": "Can you explain why multiprocessing is required for CPU-bound tasks in Python due to the GIL while threading works for IO bound?",
        "relevant_docs": ["doc_python_concurrency_gil"],
        "relevant_chunks": ["doc_python_concurrency_gil_c0"],
        "supported": True,
        "query_type": "long"
    },

    # --- Data Science (7 questions) ---
    {
        "id": "q009",
        "category": "data_science",
        "question": "What is NumPy broadcasting and how does it work?",
        "relevant_docs": ["doc_numpy_vectorization"],
        "relevant_chunks": ["doc_numpy_vectorization_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q010",
        "category": "data_science",
        "question": "What is the difference between .loc and .iloc in Pandas?",
        "relevant_docs": ["doc_pandas_indexing_selection"],
        "relevant_chunks": ["doc_pandas_indexing_selection_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q011",
        "category": "data_science",
        "question": "Why does chained assignment trigger SettingWithCopyWarning in Pandas?",
        "relevant_docs": ["doc_pandas_indexing_selection"],
        "relevant_chunks": ["doc_pandas_indexing_selection_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q012",
        "category": "data_science",
        "question": "How does the split-apply-combine strategy work in Pandas groupby?",
        "relevant_docs": ["doc_pandas_groupby_aggregations"],
        "relevant_chunks": ["doc_pandas_groupby_aggregations_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q013",
        "category": "data_science",
        "question": "How are outliers identified using the Interquartile Range (IQR)?",
        "relevant_docs": ["doc_data_cleaning_missing_outliers"],
        "relevant_chunks": ["doc_data_cleaning_missing_outliers_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q014",
        "category": "data_science",
        "question": "vectorized operations NumPy memory speedup",
        "relevant_docs": ["doc_numpy_vectorization"],
        "relevant_chunks": ["doc_numpy_vectorization_c0"],
        "supported": True,
        "query_type": "short"
    },
    {
        "id": "q015",
        "category": "data_science",
        "question": "What methods exist for imputing missing values in tabular data?",
        "relevant_docs": ["doc_data_cleaning_missing_outliers"],
        "relevant_chunks": ["doc_data_cleaning_missing_outliers_c0"],
        "supported": True,
        "query_type": "paraphrased"
    },

    # --- Machine Learning (7 questions) ---
    {
        "id": "q016",
        "category": "machine_learning",
        "question": "What is the bias-variance tradeoff in machine learning?",
        "relevant_docs": ["doc_bias_variance_tradeoff"],
        "relevant_chunks": ["doc_bias_variance_tradeoff_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q017",
        "category": "machine_learning",
        "question": "How does Stochastic Gradient Descent differ from Batch Gradient Descent?",
        "relevant_docs": ["doc_gradient_descent_optimization"],
        "relevant_chunks": ["doc_gradient_descent_optimization_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q018",
        "category": "machine_learning",
        "question": "What is the key difference between L1 Lasso and L2 Ridge regularization?",
        "relevant_docs": ["doc_regularization_l1_l2"],
        "relevant_chunks": ["doc_regularization_l1_l2_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q019",
        "category": "machine_learning",
        "question": "Why does L1 regularization cause sparsity while L2 regularization does not?",
        "relevant_docs": ["doc_regularization_l1_l2"],
        "relevant_chunks": ["doc_regularization_l1_l2_c0"],
        "supported": True,
        "query_type": "paraphrased"
    },
    {
        "id": "q020",
        "category": "machine_learning",
        "question": "How do Random Forests build upon Bagging to reduce variance?",
        "relevant_docs": ["doc_ensemble_tree_methods"],
        "relevant_chunks": ["doc_ensemble_tree_methods_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q021",
        "category": "machine_learning",
        "question": "Bias variance",
        "relevant_docs": ["doc_bias_variance_tradeoff"],
        "relevant_chunks": ["doc_bias_variance_tradeoff_c0"],
        "supported": True,
        "query_type": "short"
    },
    {
        "id": "q022",
        "category": "machine_learning",
        "question": "Explain how boosting sequentially trains weak learners to correct previous residual errors.",
        "relevant_docs": ["doc_ensemble_tree_methods"],
        "relevant_chunks": ["doc_ensemble_tree_methods_c0"],
        "supported": True,
        "query_type": "long"
    },

    # --- Deep Learning (6 questions) ---
    {
        "id": "q023",
        "category": "deep_learning",
        "question": "How does the backpropagation algorithm compute gradients in neural networks?",
        "relevant_docs": ["doc_neural_networks_backpropagation"],
        "relevant_chunks": ["doc_neural_networks_backpropagation_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q024",
        "category": "deep_learning",
        "question": "Why does the ReLU activation function help prevent vanishing gradients?",
        "relevant_docs": ["doc_activations_vanishing_gradients"],
        "relevant_chunks": ["doc_activations_vanishing_gradients_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q025",
        "category": "deep_learning",
        "question": "What is the purpose of Batch Normalization in deep architectures?",
        "relevant_docs": ["doc_optimizers_batchnorm"],
        "relevant_chunks": ["doc_optimizers_batchnorm_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q026",
        "category": "deep_learning",
        "question": "Explain the scaled dot-product attention formula in Transformers.",
        "relevant_docs": ["doc_transformers_self_attention"],
        "relevant_chunks": ["doc_transformers_self_attention_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q027",
        "category": "deep_learning",
        "question": "Why is the dot product divided by the square root of d_k in Transformer attention?",
        "relevant_docs": ["doc_transformers_self_attention"],
        "relevant_chunks": ["doc_transformers_self_attention_c0"],
        "supported": True,
        "query_type": "paraphrased"
    },
    {
        "id": "q028",
        "category": "deep_learning",
        "question": "Adam optimizer momentum moments",
        "relevant_docs": ["doc_optimizers_batchnorm"],
        "relevant_chunks": ["doc_optimizers_batchnorm_c0"],
        "supported": True,
        "query_type": "short"
    },

    # --- Computer Vision (6 questions) ---
    {
        "id": "q029",
        "category": "computer_vision",
        "question": "How do convolutional layers achieve parameter sharing?",
        "relevant_docs": ["doc_convolutional_neural_networks"],
        "relevant_chunks": ["doc_convolutional_neural_networks_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q030",
        "category": "computer_vision",
        "question": "How do residual connections in ResNet solve the degradation problem?",
        "relevant_docs": ["doc_residual_networks"],
        "relevant_chunks": ["doc_residual_networks_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q031",
        "category": "computer_vision",
        "question": "What is the architectural difference between two-stage and one-stage object detectors?",
        "relevant_docs": ["doc_object_detection_architectures"],
        "relevant_chunks": ["doc_object_detection_architectures_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q032",
        "category": "computer_vision",
        "question": "How does Non-Maximum Suppression (NMS) eliminate redundant bounding boxes?",
        "relevant_docs": ["doc_detection_metrics_nms"],
        "relevant_chunks": ["doc_detection_metrics_nms_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q033",
        "category": "computer_vision",
        "question": "What is Intersection over Union (IoU) in object detection?",
        "relevant_docs": ["doc_detection_metrics_nms"],
        "relevant_chunks": ["doc_detection_metrics_nms_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q034",
        "category": "computer_vision",
        "question": "YOLO vs Faster R-CNN detection speed",
        "relevant_docs": ["doc_object_detection_architectures"],
        "relevant_chunks": ["doc_object_detection_architectures_c0"],
        "supported": True,
        "query_type": "short"
    },

    # --- NLP (6 questions) ---
    {
        "id": "q035",
        "category": "nlp",
        "question": "What are subword tokenization algorithms like BPE and WordPiece?",
        "relevant_docs": ["doc_tokenization_subword_algorithms"],
        "relevant_chunks": ["doc_tokenization_subword_algorithms_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q036",
        "category": "nlp",
        "question": "How is TF-IDF calculated and what does it measure?",
        "relevant_docs": ["doc_statistical_nlp_tfidf"],
        "relevant_chunks": ["doc_statistical_nlp_tfidf_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q037",
        "category": "nlp",
        "question": "What is the difference between CBOW and Skip-gram in Word2Vec?",
        "relevant_docs": ["doc_dense_embeddings_word2vec"],
        "relevant_chunks": ["doc_dense_embeddings_word2vec_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q038",
        "category": "nlp",
        "question": "What is Retrieval-Augmented Generation (RAG) and why is it needed?",
        "relevant_docs": ["doc_retrieval_augmented_generation"],
        "relevant_chunks": ["doc_retrieval_augmented_generation_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q039",
        "category": "nlp",
        "question": "How does RAG mitigate hallucination in large language models?",
        "relevant_docs": ["doc_retrieval_augmented_generation"],
        "relevant_chunks": ["doc_retrieval_augmented_generation_c0"],
        "supported": True,
        "query_type": "paraphrased"
    },
    {
        "id": "q040",
        "category": "nlp",
        "question": "TF-IDF formula inverse document frequency",
        "relevant_docs": ["doc_statistical_nlp_tfidf"],
        "relevant_chunks": ["doc_statistical_nlp_tfidf_c0"],
        "supported": True,
        "query_type": "short"
    },

    # --- DBMS (8 questions) ---
    {
        "id": "q041",
        "category": "dbms",
        "question": "What are the ACID properties in database transactions?",
        "relevant_docs": ["doc_acid_transactions_wal"],
        "relevant_chunks": ["doc_acid_transactions_wal_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q042",
        "category": "dbms",
        "question": "How does Write-Ahead Logging (WAL) guarantee durability?",
        "relevant_docs": ["doc_acid_transactions_wal"],
        "relevant_chunks": ["doc_acid_transactions_wal_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q043",
        "category": "dbms",
        "question": "What is Third Normal Form (3NF) in database normalization?",
        "relevant_docs": ["doc_relational_normalization"],
        "relevant_chunks": ["doc_relational_normalization_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q044",
        "category": "dbms",
        "question": "Why are B+Trees preferred over standard B-Trees for database indexing?",
        "relevant_docs": ["doc_database_indexes_btrees"],
        "relevant_chunks": ["doc_database_indexes_btrees_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q045",
        "category": "dbms",
        "question": "What is the difference between a Hash Join and a Nested Loop Join?",
        "relevant_docs": ["doc_sql_joins_query_optimization"],
        "relevant_chunks": ["doc_sql_joins_query_optimization_c0"],
        "supported": True,
        "query_type": "exact"
    },
    {
        "id": "q046",
        "category": "dbms",
        "question": "Explain the difference between 2NF and 3NF regarding partial vs transitive dependencies.",
        "relevant_docs": ["doc_relational_normalization"],
        "relevant_chunks": ["doc_relational_normalization_c0"],
        "supported": True,
        "query_type": "paraphrased"
    },
    {
        "id": "q047",
        "category": "dbms",
        "question": "3NF database normalization",
        "relevant_docs": ["doc_relational_normalization"],
        "relevant_chunks": ["doc_relational_normalization_c0"],
        "supported": True,
        "query_type": "short"
    },
    {
        "id": "q048",
        "category": "dbms",
        "question": "Can you elaborate on how relational database query optimizers choose between hash joins and merge joins?",
        "relevant_docs": ["doc_sql_joins_query_optimization"],
        "relevant_chunks": ["doc_sql_joins_query_optimization_c0"],
        "supported": True,
        "query_type": "long"
    },

    # --- Out-of-Domain & Unsupported Questions (12 questions) ---
    {
        "id": "q049",
        "category": "general",
        "question": "Who won the Martian Olympics 100m sprint in the year 2099?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q050",
        "category": "general",
        "question": "What is the attendance policy of the Hogwarts School of Witchcraft and Wizardry?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q051",
        "category": "general",
        "question": "How do you manufacture vibranium armor using quantum plasma torches?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q052",
        "category": "general",
        "question": "What is the chemical composition of spice melange on planet Arrakis?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q053",
        "category": "general",
        "question": "Explain the warp drive antimatter dilution ratio for starships.",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q054",
        "category": "general",
        "question": "Who was elected Emperor of Antarctica in 1912?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q055",
        "category": "general",
        "question": "What is the exact recipe for Pan-Galactic Gargle Blaster in Python?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q056",
        "category": "general",
        "question": "Which programming language was created by subterranean Mole people in 1845?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q057",
        "category": "general",
        "question": "What are the tax filing rules for time travelers visiting ancient Rome?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q058",
        "category": "general",
        "question": "Explain the hyperspace navigation protocol for Cygnus space freighters.",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q059",
        "category": "general",
        "question": "Who holds the world record for breathing underwater on Jupiter?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    },
    {
        "id": "q060",
        "category": "general",
        "question": "What is the internal clock speed of a positronic brain in Cybertron robots?",
        "relevant_docs": [],
        "relevant_chunks": [],
        "supported": False,
        "query_type": "out_of_domain"
    }
]


def generate_all():
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean previous old single-file structure if present
    for old_file in raw_dir.rglob("*.md"):
        try:
            old_file.unlink()
        except Exception:
            pass

    # 2. Write 30 new documents
    for rel_path, text in DOCUMENTS.items():
        out_path = raw_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        print(f"  [Created Document] {out_path.name}")

    # 3. Write evaluation dataset
    eval_file = eval_dir / "eval_questions.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(EVAL_QUESTIONS, f, indent=2)
    print(f"  [Created Eval Dataset] {eval_file.name} ({len(EVAL_QUESTIONS)} questions)")


if __name__ == "__main__":
    generate_all()
