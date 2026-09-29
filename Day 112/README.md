# 🧠 Day 112 / 200: MiniGPT & Autoregressive Transformers

> **Curriculum Progress**: Day 112 / 200 (56% Complete — 88 Days Remaining)  
> **Topic**: Generative Pre-trained Transformer (GPT), Causal Language Modeling, Pre-LN Architecture, Decoding Strategies (Greedy, Temperature, Top-K, Top-P), and Language Model Evaluation.

---

## 📖 Table of Contents
1. [Overview & Architectural Vision](#overview--architectural-vision)
2. [MiniGPT Architecture](#minigpt-architecture)
3. [Directory Structure](#directory-structure)
4. [Reproduction & Execution Guide](#reproduction--execution-guide)
5. [Empirical Results & Benchmark Summary](#empirical-results--benchmark-summary)
6. [Decoding Strategy Gallery](#decoding-strategy-gallery)
7. [Comprehensive Interview Questions & Answers (32 Q&As)](#comprehensive-interview-questions--answers-32-qas)
   - [Category A: Causal Language Modeling Foundations (Q1–Q4)](#category-a-causal-language-modeling-foundations)
   - [Category B: Transformer Architecture & Causal Masking (Q5–Q8)](#category-b-transformer-architecture--causal-masking)
   - [Category C: Attention Mechanism & Efficiency (Q9–Q12)](#category-c-attention-mechanism--efficiency)
   - [Category D: Optimization & Training Dynamics (Q13–Q16)](#category-d-optimization--training-dynamics)
   - [Category E: Autoregressive Decoding & Sampling (Q17–Q20)](#category-e-autoregressive-decoding--sampling)
   - [Category F: Metrics & Evaluation (Q21–Q24)](#category-f-metrics--evaluation)
   - [Category G: System & Production Optimization (Q25–Q28)](#category-g-system--production-optimization)
   - [Category H: Architecture Comparisons & Failure Modes (Q29–Q32)](#category-h-architecture-comparisons--failure-modes)

---

## 🚀 Overview & Architectural Vision

Day 112 focuses on **GPT (Generative Pre-trained Transformer)**: the autoregressive decoder architecture that powers modern generative AI. While Day 110 implemented the Encoder-Decoder Transformer and Day 111 built the Bidirectional Encoder (BERT), Day 112 implements the Decoder-Only paradigm.

### The Autoregressive Paradigm
$$P(x_1, x_2, \dots, x_N) = \prod_{t=1}^N P(x_t \mid x_{<t})$$

In causal language modeling, every token prediction is strictly conditioned on preceding tokens. Causal masking ensures that future tokens are completely invisible during training and inference, enabling parallel training via teacher forcing while preserving generative integrity.

---

## 🏛️ MiniGPT Architecture

```text
Input Tokens: [x_1, x_2, ..., x_t]
        │
┌───────▼────────────────────────┐
│  Token Embeddings: (V -> C)    │
│  Position Embeddings: (T -> C) │
│  Dropout (p = 0.1)             │
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐  ◄── Block 1
│  LayerNorm 1                   │
│  Causal Multi-Head Attention   │  (Lower-triangular mask M_ij = -inf)
│  Residual Connection (+)       │
│  LayerNorm 2                   │
│  GELU MLP (C -> 4C -> C)       │
│  Residual Connection (+)       │
└───────┬────────────────────────┘
        │  [Repeated across 4 Blocks]
┌───────▼────────────────────────┐
│  Final LayerNorm               │
│  LM Head (Linear: C -> V)      │  (Tied weights with Token Embeddings)
└───────┬────────────────────────┘
        ▼
Next-Token Logits: (B, T, V)
```

### Hyperparameters
- **Vocabulary Size**: 59 characters
- **Context Length ($T$)**: 64 tokens
- **Embedding Dimension ($C$)**: 128
- **Heads ($H$)**: 4 ($d_k = 32$ per head)
- **Layers ($L$)**: 4 Pre-LN blocks
- **Total Parameters**: 809,088 (weight tied)

---

## 📁 Directory Structure

```text
Day 112/
├── app/
│   ├── config.py                 # Central hyperparameter configurations
│   ├── data/
│   │   ├── dataset.py            # Corpus loader and train/val splitting
│   │   └── batching.py           # Shifted teacher-forcing mini-batch generation
│   ├── evaluation/
│   │   ├── metrics.py            # Perplexity and Distinct-N metrics
│   │   ├── repetition.py         # N-gram repetition rate and redundancy analysis
│   │   └── plots.py              # Loss curve visualization and report generators
│   ├── generation/
│   │   ├── generate.py           # Autoregressive decoding loop with sliding window
│   │   └── sampling.py           # Greedy, Temperature, Top-K, and Top-P sampling
│   ├── model/
│   │   ├── attention.py          # Multi-Head Causal Self-Attention
│   │   ├── block.py              # Pre-LN Transformer Block & GELU MLP
│   │   └── gpt.py                # Full MiniGPT Architecture with weight tying
│   ├── tokenizer/
│   │   ├── char_tokenizer.py     # Production CharacterTokenizer
│   │   └── bpe_tokenizer.py      # Simple BPE tokenizer demonstration
│   ├── training/
│   │   ├── checkpoint.py         # Model checkpoint save/load routines
│   │   ├── optimizer.py          # AdamW with weight decay decoupling
│   │   └── trainer.py            # MiniGPTTrainer with gradient clipping
│   └── main.py                   # Complete pipeline orchestrator
├── coding_challenges/            # 10 standalone runnable coding challenges
│   ├── challenge_1.py ... challenge_10.py
├── data/
│   └── input.txt                 # Shakespeare corpus excerpt (10,156 chars)
├── outputs/
│   ├── charts/loss_curves.png    # Training loss & perplexity curves
│   ├── generation_comparison.md  # Detailed qualitative decoding comparisons
│   ├── metrics.csv               # Historical training steps, loss, and perplexity
│   ├── sample_generations.txt    # Generated text across multiple prompts
│   ├── temperature_comparison.md # Temperature sensitivity benchmark table
│   └── vocab.json                # Serialized character vocabulary
├── scratch/                      # Pure Python / NumPy reference implementations
│   ├── attention.py
│   ├── causal_mask.py
│   ├── char_tokenizer.py
│   ├── generation.py
│   ├── gpt_model.py
│   ├── sampling.py
│   └── transformer_block.py
├── tests/                        # 85 comprehensive unit tests
│   ├── conftest.py
│   ├── test_attention.py
│   ├── test_generation.py
│   ├── test_mask.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_training.py
├── DAY_112_REPORT.md             # 28-section technical report
└── README.md                     # Educational overview & 32 interview Q&As
```

---

## 💻 Reproduction & Execution Guide

### 1. Run the Complete Training & Generation Pipeline
```bash
cd "Day 112"
python -m app.main
```
Outputs are written directly to `Day 112/outputs/`.

### 2. Run the Full Test Suite
```bash
pytest tests/ -v
```
Verifies all 85 unit tests across tokenization, masking, attention, modeling, sampling, evaluation, and training.

### 3. Run Standalone Coding Challenges
```bash
python coding_challenges/challenge_1.py
# Or run all 10 challenges in sequence:
python -c "import subprocess, sys; [subprocess.run([sys.executable, f'coding_challenges/challenge_{i}.py'], check=True) for i in range(1, 11)]"
```

---

## 📊 Empirical Results & Benchmark Summary

Trained for 600 iterations on CPU (~69 seconds):

| Step | Train Loss | Val Loss | Val Perplexity | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **001** | 3.7898 | 3.8050 | 44.93 | Initial random initialization |
| **100** | 2.4613 | 2.5921 | 13.36 | Learns basic whitespace and vowel distributions |
| **300** | 2.1689 | 2.3583 | 10.57 | Learns common English words and character pairings |
| **600** | 1.8517 | 2.2879 | **9.85** | Shakespearean dialogue markup and rhythm stabilized |

---

## 🎨 Decoding Strategy Gallery

Prompt: `"First Citizen:\n"`

- **Greedy Decoding ($T = 0.0$)**:
  > `First Citizen:\nhe the she she she she she she she she she she she she...`  
  *Trapped in local argmax cycle; high repetition.*

- **Temperature Sampling ($T = 0.7$)**:
  > `First Citizen:\nthe core speak,\nAll the pook him are the pround:\nWe hat I have the cour him to...`  
  *Natural Shakespearean cadence and dialogue markup without repetition loops.*

- **Top-K Sampling ($K = 10, T = 0.8$)**:
  > `First Citizen:\nthe can your him our him are the pround the say the pround the say to the speak,\nAll:...`  
  *Eliminates unusual tail characters while preserving dramatic verse.*

- **Top-P (Nucleus) Sampling ($P = 0.9, T = 0.8$)**:
  > `First Citizen:\nthe proud to your our hard to the speak:\nWe have him are to the country him are the pround:...`  
  *Dynamic candidate pool; best overall balance of grammatical flow and vocabulary diversity.*

---

## 🎓 Comprehensive Interview Questions & Answers (32 Q&As)

### Category A: Causal Language Modeling Foundations

#### Q1: What is causal language modeling and how does it differ from masked language modeling?
**Answer**:
Causal Language Modeling (CLM) optimizes next-token prediction:
$$P(X) = \prod_{t=1}^N P(x_t \mid x_1, \dots, x_{t-1})$$
At position $t$, the model is strictly constrained to attend only to tokens at positions $j \le t$. In contrast, Masked Language Modeling (MLM, used in BERT) randomly masks a subset of tokens (typically 15%) and trains the model to reconstruct them using bidirectional context (attending simultaneously to both past and future tokens). While MLM is ideal for sentence classification and extraction, it cannot naturally generate open-ended text because predicting $x_{t+1}$ would create circular information leakage during autoregressive decoding.

#### Q2: Why can't GPT attend to future tokens during training and generation?
**Answer**:
If GPT could attend to future tokens during training, predicting the next token $x_{t+1}$ at position $t$ would be trivial: the attention layer would simply look ahead at index $t+1$ and copy the ground truth token directly, reducing the training loss to zero without learning any semantic representations. During generation, future tokens do not yet exist; thus, allowing future attention during training would create a catastrophic train-test mismatch (exposure bias) where the model depends on representations that cannot exist at test time.

#### Q3: How does the probabilistic chain rule justify next-token prediction as a general language modeling objective?
**Answer**:
By the chain rule of probability:
$$P(x_1, x_2, \dots, x_N) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_N \mid x_1, \dots, x_{N-1})$$
This decomposition is exact and holds for any joint probability distribution without making any Markovian independence assumptions. Therefore, optimizing next-token conditional distributions over large diverse corpora is theoretically equivalent to learning the true joint probability distribution over all valid sequences in the language.

#### Q4: What is teacher forcing in autoregressive models, and why does it enable parallel training?
**Answer**:
Teacher forcing is a training strategy where ground truth tokens from the training sequence are fed as inputs at every position $t$, rather than feeding the model's own sampled predictions from step $t-1$. Because ground truth prefixes are known in advance and causal masking prevents information leakage from the future, all $T$ token predictions across the sequence can be computed simultaneously in a single forward pass with standard matrix multiplications. This eliminates the $\mathcal{O}(T)$ sequential unrolling required by RNNs and enables massive GPU parallelism.

---

### Category B: Transformer Architecture & Causal Masking

#### Q5: Explain the mathematical operation behind the causal attention mask ($M_{ij} = -\infty$ for $j > i$).
**Answer**:
The attention score matrix $\mathbf{S} \in \mathbb{R}^{T \times T}$ contains pairwise dot products $S_{ij} = \frac{\mathbf{q}_i \mathbf{k}_j^\top}{\sqrt{d_k}}$. An additive mask $\mathbf{M}$ is defined with $M_{ij} = 0.0$ for $j \le i$ and $M_{ij} = -\infty$ for $j > i$.
When softmax is applied along each row:
$$A_{ij} = \frac{\exp(S_{ij} + M_{ij})}{\sum_{k=1}^T \exp(S_{ik} + M_{ik})}$$
For any future position $j > i$, $\exp(S_{ij} - \infty) = 0$. Consequently, the attention weight $A_{ij}$ is exactly $0.0$, completely blocking gradient flow and information transfer from position $j$ to position $i$.

#### Q6: What is Pre-Layer Normalization (Pre-LN) and why did modern GPT models switch from Post-LN to Pre-LN?
**Answer**:
In Post-LN (Vaswani et al., 2017):
$$\mathbf{x}^{(l)} = \text{LayerNorm}(\mathbf{x}^{(l-1)} + \text{SubLayer}(\mathbf{x}^{(l-1)}))$$
Here, LayerNorm normalizes the sum, attenuating gradient magnitude as backpropagation passes through deeper layers. This required strict learning rate warm-up schedules to avoid early divergence.
In Pre-LN (Radford et al., 2019):
$$\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}^{(l-1)}))$$
Normalization is applied on the sub-layer branches prior to attention and MLP. The identity residual connection $\mathbf{x}^{(l-1)}$ passes straight through unnormalized, establishing an unimpeded gradient highway:
$$\frac{\partial \mathbf{x}^{(L)}}{\partial \mathbf{x}^{(0)}} = \mathbf{I} + \sum_{l=1}^L \frac{\partial \text{SubLayer}_l}{\partial \mathbf{x}^{(l-1)}}$$
This stabilizes training dynamics and allows scaling to hundreds of layers without warmup instability.

#### Q7: Why does MiniGPT use GELU instead of ReLU in its feed-forward network?
**Answer**:
ReLU is defined as $\max(0, x)$. It has a hard mathematical discontinuity in its derivative at $x = 0$ and completely sets gradients to zero for all negative activations ($x < 0$). If a neuron's weights shift such that it always produces negative activations, it ceases to receive gradients and dies ("dying ReLU").
GELU (Gaussian Error Linear Unit) weights inputs by their percentile in a standard normal distribution:
$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$
GELU is smooth, non-monotonic, and retains small negative gradients for negative inputs, enabling continuous learning and superior empirical convergence in language models.

#### Q8: How does learned absolute positional embedding differ from sinusoidal encoding?
**Answer**:
Sinusoidal positional encoding uses fixed trigonometric functions of deterministic frequencies:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$
It requires zero learned parameters and theoretically generalizes to arbitrary sequence lengths.
Learned absolute positional embeddings allocate a trainable weight matrix $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times C}$. The model updates these vectors via backpropagation, learning dataset-specific spatial relationships. However, learned embeddings strictly cannot process sequences longer than $T_{\text{max}}$ without interpolation or architecture extensions.

---

### Category C: Attention Mechanism & Efficiency

#### Q9: Walk through the multi-head self-attention projection and splitting tensor dimensions.
**Answer**:
1. Input tensor: $\mathbf{X} \in \mathbb{R}^{B \times T \times C}$.
2. Fused Linear Projection (`c_attn`): $\mathbf{W} \in \mathbb{R}^{C \times 3C}$ maps $\mathbf{X}$ to $\mathbf{QKV} \in \mathbb{R}^{B \times T \times 3C}$.
3. Chunking: Split into Query, Key, and Value tensors, each $\mathbb{R}^{B \times T \times C}$.
4. Reshaping for $H$ heads with $d_k = C / H$: Reshape to $(B, T, H, d_k)$ and transpose dimensions 1 and 2 to yield $(B, H, T, d_k)$.
5. Scaled Batched Dot-Product:
   $$\mathbf{S} = (\mathbf{Q} \mathbf{K}^\top) / \sqrt{d_k} \in \mathbb{R}^{B \times H \times T \times T}$$
6. Causal Mask & Softmax: Apply lower-triangular mask and row-wise softmax to get weights $(B, H, T, T)$.
7. Value Multiplication: $\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{B \times H \times T \times d_k}$.
8. Output Fusion: Transpose back to $(B, T, H, d_k)$, reshape contiguously to $(B, T, C)$, and pass through linear projection $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{C \times C}$.

#### Q10: Why do we scale the dot product by $1/\sqrt{d_k}$ in scaled dot-product attention?
**Answer**:
Assuming the components of Query $\mathbf{q}$ and Key $\mathbf{k}$ are independent random variables with zero mean and unit variance ($\mathbb{E}[q_i] = 0, \text{Var}(q_i) = 1$):
$$\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$
The mean is $\mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0$ and the variance is:
$$\text{Var}(\mathbf{q} \cdot \mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
For large $d_k$ (e.g., $d_k = 64$ or $128$), the dot products grow large in magnitude ($|\mathbf{q} \cdot \mathbf{k}| \sim \sqrt{d_k}$). Large logits push the softmax function into regions with near-zero gradients (saturation), leading to vanishing gradients. Dividing by $\sqrt{d_k}$ normalizes the variance back to $1.0$, preserving healthy gradient flow.

#### Q11: What is the computational complexity of standard self-attention, and where is the primary memory bottleneck?
**Answer**:
- **Time Complexity**: $\mathcal{O}(B \cdot T^2 \cdot C)$. The matrix multiplication of $\mathbf{Q} \in \mathbb{R}^{B \times H \times T \times d_k}$ with $\mathbf{K}^\top \in \mathbb{R}^{B \times H \times d_k \times T}$ takes $\mathcal{O}(T^2 \cdot d_k)$ operations per head, totaling $\mathcal{O}(T^2 \cdot C)$ across all $H$ heads.
- **Space Bottleneck**: Materializing the intermediate attention score matrix $\mathbf{S} \in \mathbb{R}^{B \times H \times T \times T}$ in GPU memory for backpropagation requires $\mathcal{O}(B \cdot H \cdot T^2)$ bytes. For long sequences ($T = 32\text{k}$), this quadratic memory consumption causes Out-Of-Memory (OOM) errors.

#### Q12: How does causal masking affect GPU memory access patterns during training vs generation?
**Answer**:
During **training**, the entire $T \times T$ lower-triangular matrix is processed simultaneously using dense tensor matrix-multiplication kernels (GEMM). Because the upper triangle is filled with $-\infty$, roughly 50% of the computation involves masked-out values, but dense GEMM executes at peak hardware FLOP efficiency.
During **autoregressive generation**, only a single new token query $\mathbf{q}_{T+1} \in \mathbb{R}^{1 \times C}$ is multiplied against the historical keys $\mathbf{K}_{\text{past}} \in \mathbb{R}^{T \times C}$. The operation is matrix-vector multiplication (GEMV), which is memory-bandwidth bound rather than compute bound.

---

### Category D: Optimization & Training Dynamics

#### Q13: What is weight tying, why is it used, and how does it affect model parameter counts and gradient updates?
**Answer**:
Weight tying sets the embedding lookup table and the output projection matrix to share the identical parameter tensor in memory: $\mathbf{W}_{\text{unembed}} = \mathbf{W}_{\text{embed}} \in \mathbb{R}^{|V| \times C}$.
- **Benefits**: Reduces total model parameter count by $|V| \times C$ (often 20–30% of total parameters in small-to-medium models). It enforces semantic duality: tokens with similar vector representations in input space have similar output logit projections.
- **Gradients**: During backpropagation, $\mathbf{W}_{\text{embed}}$ accumulates gradients from both the input embedding layer and the output projection:
  $$\nabla_{\mathbf{W}} \mathcal{L} = \nabla_{\mathbf{W}_{\text{embed}}} \mathcal{L} + \nabla_{\mathbf{W}_{\text{unembed}}} \mathcal{L}$$

#### Q14: Why do we decouple weight decay in AdamW so that 1D biases and LayerNorm parameters receive zero decay?
**Answer**:
Weight decay applies an L2 penalty encouraging weight vectors toward smaller Euclidean norms, which prevents overfitting in high-dimensional linear projections.
However, 1D parameters (biases $\mathbf{b}$ and LayerNorm scaling factors $\gamma$) control output translation and variance calibration rather than capacity. Decaying LayerNorm gains toward zero suppresses layer representations and destabilizes gradient scaling. AdamW decouples weight decay from gradient updates, applying decay strictly to 2D weight matrices ($\mathbf{W} \leftarrow \mathbf{W}(1 - \lambda \cdot \eta)$) while omitting 1D tensors.

#### Q15: Why is gradient clipping critical when training Transformer models?
**Answer**:
Self-attention layers and residual streams can occasionally generate abrupt gradient spikes when sequence tokens interact anomalously. If an unclipped gradient update is applied:
$$\theta \leftarrow \theta - \eta \cdot \mathbf{g}$$
a massive $\|\mathbf{g}\|_2$ can catapult weights into suboptimal parameter space, causing loss spikes or numerical overflow (`NaN`). Gradient clipping scales the gradient vector down if its global L2 norm exceeds threshold $c$:
$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{c}{\|\mathbf{g}\|_2}\right)$$
This preserves the direction of the gradient vector while strictly bounding step size.

#### Q16: How does the training context window limit differ from the inference sequence length?
**Answer**:
The training context window $T_{\text{train}}$ is the maximum number of consecutive tokens packed into a single training sample (e.g., $T_{\text{train}} = 64$ in MiniGPT).
At inference time, an autoregressive model can generate arbitrary numbers of tokens ($M \gg T_{\text{train}}$) by utilizing a sliding context window:
$$\mathbf{x}_{\text{cond}} = \mathbf{x}[:, -T_{\text{train}}:]$$
The model continually conditions on the most recent $T_{\text{train}}$ tokens, discarding tokens that have fallen outside the receptive window.

---

### Category E: Autoregressive Decoding & Sampling

#### Q17: Why does greedy decoding frequently lead to repetitive loops in open-ended text generation?
**Answer**:
Greedy decoding selects $x_t = \arg\max_{v} P(v \mid x_{<t})$ at every step. Because greedy decoding optimizes locally without exploring alternative paths:
1. It frequently enters states where common phrases (e.g., "in order to") increase the probability of their own successors, forming deterministic closed cycles ($A \to B \to C \to A$).
2. Natural human language does not consistently pick the single highest-probability word; human text meanders across medium-probability tokens to convey information. Greedy decoding collapses variance, yielding monotonous, generic, and looping text.

#### Q18: Explain the mathematical effect of temperature scaling on the logit distribution.
**Answer**:
Temperature scaling divides logits by parameter $\tau > 0$ before softmax:
$$P_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$
- **$\tau < 1.0$ (Cold)**: Amplifies differences between logits ($z_i / \tau$). Highest logits dominate exponentially, sharpening the distribution toward a peaky, low-entropy argmax distribution.
- **$\tau = 1.0$**: Preserves the model's raw uncalibrated probabilities.
- **$\tau > 1.0$ (Warm)**: Contracts differences between logits. As $\tau \to \infty$, $z_i / \tau \to 0$, causing probabilities to approach a uniform distribution ($1 / |V|$), maximizing Shannon entropy and diversity at the cost of coherence.

#### Q19: How does top-k truncation sampling work, and what is its primary limitation?
**Answer**:
Top-K sampling identifies the $K$ largest logits in $\mathbf{z}$, sets all remaining logits to $-\infty$, and samples from the renormalized softmax distribution across those $K$ tokens.
- **Limitation**: $K$ is static and context-agnostic.
  - When the model is confident (e.g., predicting the word after "artificial"), only 1 or 2 tokens make sense ("intelligence"). A fixed $K = 50$ forces the model to retain 48 implausible tokens in the sampling pool.
  - When the model is genuinely uncertain, dozens of words might be valid, but fixed $K = 10$ prematurely cuts off legitimate creative continuations.

#### Q20: How does top-p (nucleus) sampling solve the fixed-k problem in dynamic context scenarios?
**Answer**:
Top-P (nucleus) sampling dynamically bounds the candidate pool based on cumulative probability mass:
$$V^{(P)} = \left\{ v \in V : \sum_{i \in V^{(P)}} P(x_i) \ge P \right\}$$
- When the distribution is concentrated (low entropy), $V^{(P)}$ automatically shrinks (sometimes to a single token).
- When the distribution is flat (high entropy), $V^{(P)}$ automatically expands to include all viable candidates.
This provides adaptive tail truncation without manual per-context tuning.

---

### Category F: Metrics & Evaluation

#### Q21: Define perplexity mathematically and explain why it represents the effective branching factor.
**Answer**:
Perplexity is the exponentiated average negative log-likelihood (cross-entropy loss $\mathcal{L}$):
$$\text{PPL} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{N} \sum_{t=1}^N \log P(x_t \mid x_{<t})\right)$$
If a model has a uniform probability distribution over $K$ choices at each step, its cross-entropy loss is $\mathcal{L} = -\log(1/K) = \log K$, and its perplexity is $\exp(\log K) = K$.
Therefore, a perplexity of $9.85$ indicates that on average, the model's uncertainty at each step is equivalent to selecting uniformly among $9.85$ equally plausible candidate tokens.

#### Q22: Why is cross-entropy loss directly proportional to log-perplexity?
**Answer**:
Taking the natural logarithm of perplexity:
$$\ln(\text{PPL}) = \ln(\exp(\mathcal{L})) = \mathcal{L}$$
Thus, minimizing cross-entropy loss $\mathcal{L}$ is mathematically identical to minimizing perplexity. A monotonic decrease in validation loss strictly guarantees a monotonic decrease in validation perplexity.

#### Q23: How do Distinct-1 and Distinct-2 quantify lexical and phrase diversity in generated text?
**Answer**:
$$\text{Distinct-}N = \frac{|\text{Unique } N\text{-grams}|}{|\text{Total } N\text{-grams}|} \in [0.0, 1.0]$$
- **Distinct-1**: Measures single-token (unigram) vocabulary richness. A low Distinct-1 indicates the text relies on a tiny subset of repetitive words.
- **Distinct-2**: Measures bigram diversity. Text trapped in loops (e.g., "the king and the king and...") will exhibit high unigram diversity but severely depressed Distinct-2 scores ($< 0.5$).

#### Q24: What does n-gram repetition rate measure, and how does it correlate with generation quality?
**Answer**:
$$\text{Repetition Rate}_N = \frac{|\text{Total } N\text{-grams}| - |\text{Unique } N\text{-grams}|}{|\text{Total } N\text{-grams}|} = 1.0 - \text{Distinct-}N$$
It quantifies the exact percentage of generated text that consists of duplicated $N$-grams. High repetition rates ($\ge 0.35$ for $N=3$) correlate directly with degenerate looping, whereas healthy natural language typically maintains 3-gram repetition rates below $0.15$.

---

### Category G: System & Production Optimization

#### Q25: What is KV caching, why is it essential for efficient autoregressive inference, and what is its memory footprint?
**Answer**:
In naive autoregressive generation, generating token $t+1$ requires running self-attention over all $t$ previous tokens, recomputing $\mathbf{K}$ and $\mathbf{V}$ tensors from scratch ($\mathcal{O}(t^2)$ total work).
**KV Caching** stores the computed Key and Value tensors for all previous tokens in memory:
$$\mathbf{K}_{\text{cache}} \in \mathbb{R}^{B \times H \times t \times d_k}, \quad \mathbf{V}_{\text{cache}} \in \mathbb{R}^{B \times H \times t \times d_k}$$
At step $t+1$, the model projects only the newest token to obtain $\mathbf{q}_{t+1}, \mathbf{k}_{t+1}, \mathbf{v}_{t+1}$, appends $\mathbf{k}$ and $\mathbf{v}$ to the cache, and computes attention in $\mathcal{O}(t)$ time instead of $\mathcal{O}(t^2)$.
- **Memory Footprint**:
  $$\text{Bytes} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_k \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes\_per\_elem}$$

#### Q26: What is FlashAttention, and how does it achieve $2-4\times$ speedups without changing the underlying mathematical output?
**Answer**:
Standard attention materializes the $T \times T$ attention matrix in High-Bandwidth Memory (HBM), which is slow to read and write.
FlashAttention uses **GPU SRAM tiling** and the **online softmax algorithm**:
1. Blocks of queries, keys, and values are loaded into ultra-fast on-chip SRAM.
2. Attention scores and local softmax normalization statistics ($m(x), \ell(x)$) are computed iteratively in SRAM.
3. Intermediate results are scaled and accumulated without ever materializing the large $T \times T$ matrix in global HBM.
The mathematical outputs and gradients are bitwise exact to standard attention, but memory access is reduced from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$, yielding massive speedups.

#### Q27: Explain speculative decoding and why it accelerates generation for memory-bandwidth bound models.
**Answer**:
Large LLM generation is memory-bandwidth bound: loading billions of parameters from VRAM to compute a single token underutilizes GPU compute cores.
Speculative decoding uses a compact "draft model" (e.g., MiniGPT) to quickly autoregressively predict $K$ candidate tokens: $[x_1, \dots, x_K]$.
The large target LLM then processes all $K$ tokens in a **single parallel forward pass**, comparing its predicted probabilities against the draft model's choices. Valid tokens are accepted in parallel, while rejected tokens trigger a single resample. This yields $2-3\times$ speedups while provably preserving the exact output distribution of the large model.

#### Q28: How does context window sliding handle generation requests that exceed the model's maximum position embedding limit?
**Answer**:
When sequence length exceeds the maximum context length $T$:
```python
idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
```
The model slices only the trailing $T$ tokens as input. This guarantees that position indices passed to `position_embeddings` never exceed $T - 1$, preventing index-out-of-bounds exceptions while allowing arbitrarily long generations.

---

### Category H: Architecture Comparisons & Failure Modes

#### Q29: Compare BERT, GPT, and the original Transformer across attention masking, training objectives, and primary use cases.
**Answer**:
1. **Vanilla Transformer (Day 110)**:
   - *Architecture*: Encoder-Decoder.
   - *Masking*: None in encoder; Causal + Cross-Attention in decoder.
   - *Objective*: Sequence transduction (machine translation, summarization).
2. **BERT (Day 111)**:
   - *Architecture*: Encoder Only.
   - *Masking*: Padding mask only (fully bidirectional).
   - *Objective*: Masked Language Modeling (reconstruct 15% masked tokens) + Next Sentence Prediction.
   - *Use Case*: Sentence classification, token tagging, information retrieval.
3. **GPT (Day 112)**:
   - *Architecture*: Decoder Only.
   - *Masking*: Lower-triangular causal mask ($M_{ij} = -\infty$ for $j > i$).
   - *Objective*: Causal Language Modeling (predict $x_{t+1}$ across 100% of tokens).
   - *Use Case*: Open-ended text generation, dialogue, reasoning, code generation.

#### Q30: What causes hallucination in autoregressive language models, and how do modern architectures mitigate it?
**Answer**:
- **Causes**: Autoregressive models optimize surface statistical coherence and next-token likelihood under teacher forcing. They lack an explicit ground truth verification mechanism or external world state. When context is ambiguous, the model samples tokens that are statistically plausible within the training distribution, regardless of factual accuracy.
- **Mitigations**:
  1. *Retrieval-Augmented Generation (RAG)*: Grounding model context with verified documents retrieved dynamically from external databases.
  2. *RLHF / DPO*: Aligning models using human preference datasets that penalize inaccurate claims.
  3. *Chain-of-Thought (CoT)*: Encouraging intermediate step-by-step reasoning tokens before generating the final conclusion.

#### Q31: What is the difference between character-level, subword (BPE/WordPiece), and word-level tokenization in terms of vocabulary size and sequence length?
**Answer**:
- **Word-Level**: Very large vocabulary ($|V| \ge 100{,}000$). Fails on Out-Of-Vocabulary (OOV) words, typos, and morphologically rich languages. Shortest sequence lengths.
- **Character-Level**: Tiny vocabulary ($|V| \approx 50 - 256$). Zero OOV issues. However, sequences are extremely long ($4-5\times$ longer than word-level), causing self-attention memory to scale quadratically ($\mathcal{O}(T^2)$).
- **Subword (BPE / WordPiece)**: The gold standard. Balanced vocabulary ($|V| \approx 32{,}000 - 100{,}000$). High-frequency words remain whole tokens, while rare words and typos are decomposed into character n-gram chunks. Achieves optimal trade-off between sequence length and vocabulary memory.

#### Q32: When should an ML engineer select an encoder-only model (BERT) versus a decoder-only model (GPT)?
**Answer**:
- **Select Encoder-Only (BERT)**:
  - When the task is purely discriminative: document classification, sentiment analysis, named entity recognition (NER), semantic search embeddings, or re-ranking.
  - When bidirectional context over the entire input is necessary for accurate representation.
  - When latency and model footprint must be minimal (e.g., edge deployment of a 110M parameter model).
- **Select Decoder-Only (GPT)**:
  - When the task involves generating variable-length text: question answering, conversational dialogue, summarization, creative writing, or code synthesis.
  - When leveraging in-context few-shot learning or instruction-following.
  - When building generalist systems that perform multiple tasks via unified generative prompting.
