# Transformer Self-Attention and Scaled Dot-Product Mechanics

The Transformer architecture eliminates recurrence, processing entire sequences concurrently using scaled dot-product attention: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V. Query (Q), Key (K), and Value (V) matrices are linearly projected from input token representations. Dividing the dot product by sqrt(d_k) prevents large magnitudes in high-dimensional spaces from pushing the softmax function into regions with vanishingly small gradients.

Multi-head attention splits projections into multiple representation subspaces, enabling the model to attend simultaneously to syntactic, semantic, and positional dependencies across tokens. Positional encodings (sinusoidal or learned rotary RoPE embeddings) inject sequence order information into the permutation-invariant attention mechanism.
