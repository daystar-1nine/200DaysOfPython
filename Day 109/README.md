# Attention-Based SMS Spam Classification

## Day 109 / 200

An experimental NLP project exploring attention mechanisms for sequence classification.

## Models
- Simple RNN
- LSTM
- GRU
- GRU + Attention
- TF-IDF + Logistic Regression

## Concepts
- Query
- Key
- Value
- Dot-product attention
- Scaled dot-product attention
- Self-attention
- Attention masking
- Attention visualization

## Architecture
```text
Input
  ↓
Embedding
  ↓
GRU(return_sequences=True)
  ↓
Self-Attention / Mask-Aware AttentionLayer
  ↓
Context Vector
  ↓
Dropout
  ↓
Dense(ReLU)
  ↓
Dense(Sigmoid)
```

## Evaluation
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Average Precision
- Confusion Matrix

## Analysis
- Attention heatmaps
- Attention entropy
- Error analysis
- Threshold analysis
- Parameter comparison
- Training-time comparison

## Reproducibility
All experiments use controlled data splits and fixed seeds (`42`) where possible.
Executed natively using PyTorch on Python 3.14.4.

## Important Note
Attention weights are treated as model behavior signals rather than definitive causal explanations.

---

## 💡 20 Interview Questions & Answers

### Beginner
1. **What is attention?**  
   A mechanism that allows neural networks to dynamically assign different importance weights to different parts of an input sequence when computing a representation.
2. **Why do we need attention?**  
   Recurrent networks force information sequentially into a fixed-length hidden state, creating an information bottleneck. Attention allows direct access to all tokens regardless of distance.
3. **What are Query, Key, and Value?**  
   - **Query ($Q$):** What information is being searched for.  
   - **Key ($K$):** What content or label each token represents.  
   - **Value ($V$):** The actual information retrieved and weighted by similarity between Query and Key.
4. **What is an attention score?**  
   A scalar value measuring compatibility or relevance between a query vector and a key vector (e.g. $Q K^T$).
5. **Why is softmax used in attention?**  
   It converts unbounded attention scores into non-negative probability weights that sum to 1.0, enabling a convex combination (weighted average) of values.
6. **What is self-attention?**  
   An attention mechanism where Query, Key, and Value all originate from the same input sequence, allowing tokens within the same sequence to relate to each other.

### Intermediate
7. **Explain scaled dot-product attention.**  
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$  
   Compatibility is computed via dot products, scaled by the square root of the key dimension, normalized by softmax, and multiplied by the value matrix.
8. **Why divide by $\sqrt{d_k}$?**  
   For large key dimensions $d_k$, the dot products grow large in magnitude, causing softmax to yield extremely peaked distributions with near-zero gradients (saturation). Dividing by $\sqrt{d_k}$ stabilizes gradient variance to 1.
9. **What is an attention matrix?**  
   An $L \times L$ matrix in self-attention where entry $(i, j)$ represents the attention weight that token $i$ gives to token $j$.
10. **What is an attention weight?**  
    A normalized scalar value $\alpha_i \in [0, 1]$ indicating the relative contribution of value vector $V_i$ to the final context vector.
11. **Why do we need masking?**  
    Padding tokens (`<PAD>`) carry no linguistic meaning. Without masking, softmax assigns non-zero attention weights to pad tokens. Masking sets pad positions to $-1e9$ before softmax so they receive zero weight.
12. **Why does GRU need `return_sequences=True` before attention?**  
    Attention requires the full temporal sequence of hidden states $[h_1, h_2, \dots, h_T]$ across all steps to compute scores, rather than just the final summary state $h_T$.

### Advanced
13. **What is the difference between self-attention and cross-attention?**  
    In self-attention, $Q, K, V$ all originate from the same sequence. In cross-attention (e.g. in encoder-decoder models), $Q$ comes from the decoder while $K$ and $V$ come from the encoder.
14. **Why can attention process relationships between distant tokens better than RNNs?**  
    The path length between any two tokens in self-attention is $O(1)$ operations, compared to $O(T)$ sequential steps in an RNN, preventing exponential gradient decay.
15. **What is the computational complexity of self-attention?**  
    $O(T^2 \cdot d)$, where $T$ is sequence length and $d$ is dimension. Matrix multiplication $Q K^T$ scales quadratically with sequence length.
16. **Why can attention become expensive for long sequences?**  
    The $T \times T$ attention matrix requires $O(T^2)$ memory and compute, becoming prohibitive for sequences of tens of thousands of tokens without sparse or linear approximations.
17. **What is multi-head attention?**  
    Running the scaled dot-product attention multiple times in parallel with distinct learnable linear projections, allowing the model to simultaneously attend to information from different representation subspaces.
18. **Why are multiple attention heads useful?**  
    A single head averages attention across all relations; multiple heads allow one head to focus on syntactic dependencies, another on semantic proximity, and another on entity references.
19. **Why is attention not automatically a causal explanation?**  
    High attention weight denotes high feature aggregation weight, but non-linear downstream interactions, correlated features, and gradient interactions mean high attention does not prove that removing the token would change the decision.
20. **How does attention lead to Transformers?**  
    By demonstrating that self-attention alone can model sequential dependencies without recurrence, Vaswani et al. (2017) discarded recurrent connections entirely, creating the parallelizable, scalable Transformer architecture.
