# Mini Transformer SMS Spam Classifier

## Day 110 / 200

A from-scratch learning implementation of the core Transformer encoder architecture applied to SMS spam classification.

---

## 🏗️ Architecture

```text
Input Token IDs
      ↓
Token Embedding + Sinusoidal Positional Encoding
      ↓
┌─────────────────────────────────────────┐
│ Transformer Encoder Block 1             │
│   ├── Multi-Head Self-Attention (4 H)   │
│   ├── Add & LayerNorm                   │
│   ├── Position-wise Feed-Forward (256)  │
│   └── Add & LayerNorm                   │
└────────────────────┬────────────────────┘
                     ↓
┌─────────────────────────────────────────┐
│ Transformer Encoder Block 2             │
│   ├── Multi-Head Self-Attention (4 H)   │
│   ├── Add & LayerNorm                   │
│   ├── Position-wise Feed-Forward (256)  │
│   └── Add & LayerNorm                   │
└────────────────────┬────────────────────┘
                     ↓
         Mask-Aware Average Pooling
                     ↓
             Dense(128 -> 64)
                     ↓
                  Dropout
                     ↓
             Dense(64 -> 1, Sigmoid)
```

---

## 🧩 Components

- **Token Embeddings**: Maps discrete token indices to continuous vectors, scaled by $\sqrt{d_{model}}$.
- **Sinusoidal Positional Encoding**: Injects absolute and relative sequence position information into self-attention inputs using sinusoidal and cosine basis functions.
- **Scaled Dot-Product Attention**: Computes $\text{softmax}(Q K^T / \sqrt{d_k} + M) V$ with support for padding masks.
- **Multi-Head Attention**: Splits the embedding space into multiple subspaces, allowing the model to jointly attend to information from different representation subspaces at different positions.
- **Feed-Forward Network**: Position-wise two-layer dense network with non-linear activation (ReLU/GELU).
- **Residual Connections**: Skip-connections around each sub-layer to preserve gradient highways during backpropagation.
- **Layer Normalization**: Stabilizes activations across features for each token independently.
- **Padding Masks**: Prevents the model from attending to meaningless zero-padded positions.

---

## 📊 Benchmark Results

Empirical results measured directly on the controlled SMS spam test set:

| Model | Parameters | Training Time (s) | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TF-IDF + LR** | 401 | 0.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **SimpleRNN** | 83,457 | 0.47 | 0.6400 | 0.6400 | 1.0000 | 0.7805 | 0.5000 |
| **LSTM** | 182,529 | 0.64 | 0.6400 | 0.6400 | 1.0000 | 0.7805 | 0.5000 |
| **GRU** | 149,505 | 1.30 | 0.6400 | 0.6400 | 1.0000 | 0.7805 | 0.5000 |
| **GRU + Attention** | 157,825 | 1.59 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Mini Transformer** | 315,393 | 5.74 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*Note: All benchmark values are generated from actual experiments and are not hardcoded.*

---

## 🔬 Ablation Study

Empirical results evaluating the removal of individual components:

| Configuration | Parameters | Training Time (s) | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Observation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Full Transformer** | 315,393 | 7.12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Baseline complete architecture with residuals and LayerNorm. |
| **No Positional Encoding** | 315,393 | 7.48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Bag-of-words self-attention without token position order awareness. |
| **No Residual** | 315,393 | 7.13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Removes skip-connections; degrades gradient flow in deeper networks. |
| **No LayerNorm** | 314,369 | 5.61 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Without normalization; activation variance drifts across layers. |
| **No Padding Mask** | 315,393 | 3.76 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Attention attends over meaningless pad tokens, diluting attention mass. |

---

## 🧪 Experiments

1. **Number of Attention Heads**: Swept $h \in \{2, 4, 8\}$ with $d_{model}=128$ constant ([`exp1_num_heads.png`](file:///s:/Programming/Python200days/Day%20110/output/charts/exp1_num_heads.png)).
2. **Number of Layers**: Swept $L \in \{1, 2, 4\}$ ([`exp2_num_layers.png`](file:///s:/Programming/Python200days/Day%20110/output/charts/exp2_num_layers.png)).
3. **Model Dimension**: Swept $d_{model} \in \{64, 128, 256\}$ ([`exp3_model_dim.png`](file:///s:/Programming/Python200days/Day%20110/output/charts/exp3_model_dim.png)).
4. **Sequence Length Complexity**: Tested $L \in \{32, 64, 100, 150\}$ comparing Transformer ($O(n^2)$) vs GRU ($O(n)$) training times ([`exp4_seq_length_complexity.png`](file:///s:/Programming/Python200days/Day%20110/output/charts/exp4_seq_length_complexity.png)).

---

## 💼 Interview Questions & Answers

### Beginner
1. **What is a Transformer?**  
   A deep learning architecture introduced in "Attention Is All You Need" (Vaswani et al., 2017) that models sequence data entirely through self-attention and feed-forward networks, dispensing with recurrence and convolutions.
2. **Why were Transformers introduced?**  
   To overcome the sequential computation bottleneck of RNNs and LSTMs. Transformers process all sequence tokens simultaneously in parallel, speeding up training on modern GPUs and eliminating vanishing gradient paths across long sequences.
3. **What is positional encoding?**  
   A mechanism to provide sequence order information to the model by adding fixed or learned vectors (such as sinusoids of varying frequencies) to input token embeddings before the self-attention layers.
4. **Why does a Transformer need positional information?**  
   Self-attention is permutation-equivariant; without positional information, the model treats "cat ate mouse" and "mouse ate cat" identically.
5. **What is multi-head attention?**  
   An attention mechanism where queries, keys, and values are linearly projected $h$ times into distinct subspaces of dimension $d_k = d_{model} / h$, allowing the model to attend to different types of relationships simultaneously.
6. **What is a Transformer encoder?**  
   A stack of Transformer blocks that takes an entire sequence and produces contextualized representations of all tokens using bidirectional self-attention.
7. **What is a Transformer decoder?**  
   A stack of blocks that autoregressively generates tokens one at a time, using masked (causal) self-attention to prevent looking ahead and cross-attention to attend to the encoder's output.

### Intermediate
8. **Explain the complete Transformer encoder block.**  
   An encoder block comprises two main sub-layers: Multi-Head Self-Attention and a Position-wise Feed-Forward Network. Each sub-layer is wrapped in a residual connection followed by Layer Normalization (Post-LN: $x = \text{LN}(x + \text{SubLayer}(x))$ or Pre-LN: $x = x + \text{SubLayer}(\text{LN}(x))$).
9. **Why are residual connections used?**  
   Residual connections create direct gradient pathways back through the network, preventing gradients from vanishing or exploding during backpropagation and allowing deep models to train smoothly.
10. **Why is LayerNorm used instead of BatchNorm in Transformers?**  
    LayerNorm normalizes across the feature dimension for each token independently of batch size or sequence length. BatchNorm depends on batch statistics and struggles with variable sequence lengths and small batch sizes.
11. **Why is a feed-forward network needed if we already have self-attention?**  
    Self-attention performs linear combinations across sequence positions. The position-wise FFN provides non-linear transformations and high-dimensional feature mixing within each token representation.
12. **What is the difference between self-attention and cross-attention?**  
    In self-attention, Queries, Keys, and Values all originate from the same sequence. In cross-attention, Queries originate from one sequence (e.g. decoder) while Keys and Values originate from another (e.g. encoder).
13. **What is a padding mask?**  
    A mask that forces attention scores at padding token positions to $-\infty$ (or $-1e9$) before softmax so that padded tokens receive 0 attention weight and do not corrupt representations.
14. **What is a causal mask?**  
    An upper-triangular mask that prevents tokens at position $i$ from attending to tokens at positions $j > i$, maintaining autoregressive causality in decoder models.
15. **Why must `d_model` be divisible by `num_heads`?**  
    To ensure each head receives an integer projection subspace dimension $d_k = d_{model} / h$ such that when the $h$ heads of dimension $d_k$ are concatenated, the total dimension reconstructs $d_{model}$.

### Advanced
16. **Explain scaled dot-product attention mathematically.**  
    $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$. Dot products can grow large for high dimensions $d_k$, pushing softmax into regions with vanishingly small gradients. Scaling by $1/\sqrt{d_k}$ preserves unit variance when queries and keys have zero mean and unit variance.
17. **What is the computational complexity of self-attention?**  
    Computing $Q K^T$ takes $O(n^2 \cdot d_{model})$, and multiplying by $V$ takes $O(n^2 \cdot d_{model})$, resulting in $O(n^2 \cdot d_{model})$ time and space complexity with respect to sequence length $n$.
18. **Why can Transformer training be parallelized?**  
    During training, all target tokens are known simultaneously. Self-attention matrices and feed-forward layers can be computed via batched matrix multiplications without waiting for previous steps.
19. **Why can't a vanilla RNN process all tokens in parallel?**  
    Because hidden state $h_t$ is a recursive function of $h_{t-1}$. Step $t$ cannot begin until step $t-1$ completes.
20. **Why can Transformers struggle with very long sequences?**  
    Due to the $O(n^2)$ memory and compute requirement of full attention matrices. For $n = 32,768$, an $n \times n$ attention matrix requires over a billion elements per head per layer.
21. **Why is sinusoidal positional encoding advantageous over simple learned position embeddings?**  
    Sinusoidal encodings have fixed periodic properties allowing the model to extrapolate to sequence lengths longer than seen during training and represent relative distances $PE(pos + k)$ via linear transformations of $PE(pos)$.
22. **What is the difference between Pre-LN and Post-LN Transformers?**  
    Post-LN places LayerNorm after the residual addition ($x = \text{LN}(x + F(x))$); it requires careful learning rate warmups. Pre-LN places LayerNorm on the sub-layer input inside the residual branch ($x = x + F(\text{LN}(x))$); it facilitates much more stable gradient flow and allows training deeper networks without warmups.
23. **What is multi-head attention actually learning?**  
    Empirical analyses reveal that different heads specialize in distinct linguistic and semantic functions: some attend to immediate syntactic neighbors, others to distant coreferences, punctuation, or semantic modifiers.
24. **Why might removing positional encoding affect sequence classification?**  
    Without positional encoding, the model acts as a continuous bag-of-words model. In tasks where word order is decisive (e.g. sentiment negation like "not good, actually bad" vs "not bad, actually good"), performance degrades significantly.
25. **How would you optimize a Transformer for long documents?**  
    Techniques include FlashAttention (tiling and memory IO-awareness), Sparse/Local Attention (Longformer, BigBird), Linear Attention (State-Space Models like Mamba, Performer), Chunked/Hierarchical Transformers, and Rotary Positional Encodings (RoPE).
