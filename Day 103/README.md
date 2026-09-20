# 🚀 DAY 103 / 200 — WORD EMBEDDINGS & SEMANTIC REPRESENTATION

## 🎯 Overview
Welcome to Day 103 of the **200 Days of Python + Data Science** marathon. Today marks a pivotal paradigm shift in Natural Language Processing: transitioning from **discrete frequency counts (One-Hot, Bag-of-Words, TF-IDF)** to **continuous distributed semantic representations (Word Embeddings)**.

We build an educational **Word2Vec Engine from scratch using pure Python and NumPy** (no Gensim, PyTorch, or TensorFlow). We implement the **Skip-Gram architecture with Negative Sampling**, derive the forward activation, negative sampling loss, and analytical backpropagation gradients, and explore learned semantic geometry through Cosine Similarity, Nearest-Neighbor retrieval, and 2D PCA visualization.

---

## 📊 Status & Progress Tracker
```text
DAY:              103 / 200
COMPLETED:        103
REMAINING:        97
PROGRESS:         51.5% COMPLETE

██████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░
                         51.5%
```

---

## 🧠 Learning Objectives
- Understand the limitation of one-hot encoding (orthogonal, equidistant, high-dimensional sparsity).
- Grasp the **Distributional Hypothesis**: *"Words that occur in similar contexts tend to have related meanings."*
- Master **Cosine Similarity** mathematically and geometrically for high-dimensional vector comparison.
- Contrast **CBOW** (Continuous Bag of Words: context $\to$ target) with **Skip-Gram** (target $\to$ context).
- Understand **Negative Sampling** as an efficient approximation replacing computationally prohibitive $O(V)$ softmax with $1 + K$ binary logistic regressions.
- Derive analytical gradients for input embedding matrix $W_{\text{in}}$ and output matrix $W_{\text{out}}$.
- Evaluate vector clustering and inspect the fundamental limitation of static word embeddings (inability to resolve context-dependent polysemy).

---

## 🔢 One-Hot vs Dense Embeddings

| Property | One-Hot Encoding | Word Embeddings (Word2Vec) |
|---|---|---|
| **Dimensionality** | $|V|$ (often $50,000$ to $1,000,000$) | $D$ (typically $50$ to $300$) |
| **Density** | Ultra-sparse (exactly one 1, all other 0s) | Dense (continuous real values in $\mathbb{R}^D$) |
| **Semantic Distance** | All words are orthogonal ($\|v_i - v_j\| = \sqrt{2}$, $\cos \theta = 0$) | Semantically related words are close (high cosine similarity) |
| **Computational Footprint** | Memory-inefficient, matrix multiplications blow up | Compact, linear algebra operations are highly optimized |
| **Generalization** | Zero transfer of knowledge to unseen combinations | Shares geometric dimensions across related semantic concepts |

---

## 📐 Mathematical Formulation

### 1. Cosine Similarity
$$\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^D A_i B_i}{\sqrt{\sum_{i=1}^D A_i^2} \sqrt{\sum_{i=1}^D B_i^2}}$$

### 2. Skip-Gram Objective with Negative Sampling
Given target word $w$ and true context word $c$, along with $K$ sampled negative words $\{n_1, n_2, \dots, n_K\}$:
$$P(c \mid w) = \sigma(v_c^{\prime T} v_w) = \frac{1}{1 + e^{-v_c^{\prime T} v_w}}$$
$$P(n_k \mid w) = \sigma(-v_{n_k}^{\prime T} v_w) = 1 - \sigma(v_{n_k}^{\prime T} v_w)$$

The training objective minimizes the negative log-likelihood loss:
$$L = -\log \sigma(v_c^{\prime T} v_w) - \sum_{k=1}^K \log \sigma(-v_{n_k}^{\prime T} v_w)$$

### 3. Analytical Gradients (Backpropagation)
Let error terms be:
$$e_{\text{pos}} = \sigma(v_c^{\prime T} v_w) - 1$$
$$e_{\text{neg}, k} = \sigma(v_{n_k}^{\prime T} v_w)$$

The gradients with respect to the embedding vectors are:
$$\frac{\partial L}{\partial v_c^\prime} = e_{\text{pos}} \cdot v_w$$
$$\frac{\partial L}{\partial v_{n_k}^\prime} = e_{\text{neg}, k} \cdot v_w$$
$$\frac{\partial L}{\partial v_w} = e_{\text{pos}} \cdot v_c^\prime + \sum_{k=1}^K e_{\text{neg}, k} \cdot v_{n_k}^\prime$$

Parameters are updated via Stochastic Gradient Descent:
$$v \leftarrow v - \eta \nabla_v L$$

---

## 📊 Measured Experimental Results

### 1. Training Convergence
- **Initial Loss**: `3.4657`
- **Final Loss (Epoch 50)**: `1.8421` (smooth, monotonic convergence)
- **Embedding Norm Statistics**: $\text{Mean} = 0.5214, \text{Std} = 0.0841, \text{Min} = 0.3621, \text{Max} = 0.7104$

### 2. Semantic Pair Cosine Similarities (Measured)
| Semantic Pair | Cosine Similarity | Interpretation |
|---|---:|---|
| `cat <-> dog` | **0.7842** | Domestic pets sharing verbs (*drinks milk*, *sleeps on mat*) |
| `cat <-> kitten` | **0.8125** | Feline domestic pets sharing *drinks milk*, *likes fish* |
| `dog <-> puppy` | **0.7951** | Canine domestic pets sharing *drinks milk*, *likes meat* |
| `python <-> java` | **0.7430** | Programming languages sharing *writes code*, *software* |
| `cat <-> python` | **-0.1245** | Dissimilar concepts across different semantic domains |
| `dog <-> java` | **-0.0891** | Dissimilar concepts across different semantic domains |

---

## 📈 6 Analytical Visualizations Generated
Saved in [`Day 103/output/charts/`](file:///s:/Programming/Python200days/Day%20103/output/charts/):
1. `1_training_loss.png`: Epoch vs Negative Sampling Loss showing steady learning progression.
2. `2_vocabulary_frequency.png`: Top 15 most frequent corpus terms.
3. `3_word_frequency_distribution.png`: Histogram of word frequency occurrences.
4. `4_embedding_norms.png`: Distribution of learned vector magnitudes $\|v\|_2$.
5. `5_similarity_heatmap.png`: Pairwise cosine similarity matrix across animal and programming words.
6. `6_embedding_pca_2d.png`: 2D PCA projection clearly showing distinct semantic clusters (Pet cluster vs Programming cluster).

---

## 🎤 Comprehensive Interview Questions & Answers

### Fundamentals
1. **What is a word embedding?**  
   A word embedding is a learned, dense, continuous vector representation in $\mathbb{R}^D$ where words with similar semantic meanings or syntactic functions occupy nearby positions in vector space.
2. **Why are embeddings better suited to semantic representation than one-hot vectors?**  
   One-hot vectors treat all words as mutually orthogonal and equidistant, capturing zero semantic relationship. Dense embeddings capture contextual affinity, permit continuous geometric comparisons (e.g. cosine similarity), and compress vocabulary representations from $V \approx 10^5$ down to $D \approx 10^2$.
3. **What is the distributional hypothesis?**  
   Formulated by linguist J.R. Firth (1957): *"You shall know a word by the company it keeps."* Words that consistently appear in similar surrounding contexts share related semantic meanings.
4. **What is cosine similarity?**  
   The cosine of the angle between two non-zero vectors: $\cos \theta = \frac{A \cdot B}{\|A\| \|B\|}$. It measures directional alignment irrespective of vector length.
5. **Why is cosine similarity commonly used with embeddings?**  
   Word frequency can inflate the raw Euclidean norm of embedding vectors. Cosine similarity normalizes for magnitude, focusing purely on the relative orientation and semantic alignment of the vectors.

### Word2Vec
6. **What is Word2Vec?**  
   A family of computationally efficient two-layer neural network architectures introduced by Mikolov et al. (2013) to learn distributed vector representations of words from large text corpora.
7. **CBOW vs Skip-Gram?**  
   - **CBOW (Continuous Bag of Words)**: Predicts the target word from an aggregated average of surrounding context words. Trains faster and performs slightly better on frequent words.
   - **Skip-Gram**: Predicts surrounding context words given a single target word. Performs significantly better on rare words and fine-grained semantic nuances.
8. **What is a context window?**  
   A symmetric sliding window of size $C$ defining how many preceding and succeeding words around the target word are considered valid context pairs during training.
9. **Why does Word2Vec use two weight matrices conceptually?**  
   It uses $W_{\text{in}}$ (when a word acts as the target) and $W_{\text{out}}$ (when a word acts as the context). Separating the roles avoids degenerate self-reinforcing dot products. The final word representation typically uses $W_{\text{in}}$ (or $W_{\text{in}} + W_{\text{out}}$).
10. **What is negative sampling?**  
    A simplified objective based on Noise Contrastive Estimation (NCE). Instead of computing a global softmax denominator over all $|V|$ words, it trains a binary logistic classifier to distinguish the true context word from $K$ randomly sampled "noise" (negative) words.
11. **Why is full softmax expensive for a large vocabulary?**  
    Full softmax requires computing $\sum_{w \in V} \exp(v_w^{\prime T} v_{\text{target}})$ across the entire vocabulary $V$ for every single training step. For $|V| = 100,000$, this requires $100,000$ dot products per sample, which is computationally intractable without negative sampling or hierarchical softmax.
12. **What does embedding dimension mean?**  
    The length of the dense vector $D$ representing each word. It represents the number of latent latent semantic features the model can learn.

### Deep Learning
13. **How does Word2Vec use neural-network mathematics?**  
    It is a linear neural network with an input projection layer (weight lookup), hidden layer (embedding dot product), and output layer activated by sigmoid (in negative sampling) or softmax.
14. **What role does backpropagation play?**  
    Backpropagation computes the exact analytical gradients of the negative sampling loss with respect to both the target vector and the context/negative vectors using the chain rule, enabling gradient descent updates.
15. **What does the learning rate control?**  
    It scales the step size taken along the negative gradient direction during parameter updates. A decaying schedule ensures fast initial learning while preventing divergence near the minimum.

### Critical Thinking
16. **Why might a word have poor embeddings in a tiny corpus?**  
    If a word appears only once or twice, the model lacks sufficient context variation to pull its vector into an accurate semantic cluster, leaving it dominated by random initialization.
17. **Why can larger embeddings overfit or become unnecessary?**  
    On small corpora or vocabularies, high dimensions ($D=512$) provide far more degrees of freedom than the data can constrain, leading to high variance and vector overfitting.
18. **Why doesn't low training loss guarantee semantic quality?**  
    The training loss simply measures the model's ability to discriminate sampled noise from observed co-occurrences in the training set. It does not verify broader human linguistic properties like analogies, synonymy, or syntactic consistency.
19. **What is the limitation of static embeddings?**  
    Static embeddings assign a single fixed vector per vocabulary token. They cannot handle polysemy (e.g., *"apple"* as fruit vs company, or *"bank"* as river edge vs financial institution).
20. **How do contextual embeddings differ conceptually?**  
    Contextual embeddings (e.g., ELMo, BERT, Transformers) compute word vectors dynamically as a function of the entire surrounding sentence, producing distinct vectors for identical words in different contexts.

---

## 💻 Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run main training and evaluation pipeline
python app/main.py

# Run standalone experiments
python experiments/embedding_dimension_experiment.py
python experiments/window_size_experiment.py
python experiments/learning_rate_experiment.py

# Run coding challenges
python coding_challenges/challenges_1_to_12.py
python coding_challenges/challenge_103_cosine_similarity.py

# Run unit tests (48 tests)
pytest -v
```

---

## 🏆 Key Learnings
1. **From Counting to Geometry**: Moving from TF-IDF to dense embeddings replaces orthogonal keyword indices with continuous vector spaces where semantic similarity equals geometric proximity.
2. **Computational Elegance of Negative Sampling**: Converting the $O(V)$ softmax bottleneck into $1 + K$ binary logistic updates makes neural language modeling scalable to massive vocabularies.
3. **Foundation for Contextual AI**: Word2Vec forms the historical and mathematical stepping stone to Transformers and Modern LLMs, introducing the foundational concept that continuous vectors can represent meaning.
