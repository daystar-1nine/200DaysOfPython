# Transformer Scaled Dot-Product Attention Mechanisms

The core innovation of the Transformer architecture is Scaled Dot-Product Attention, computed as:
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
where $Q$ represents Queries, $K$ represents Keys, $V$ represents Values, and $d_k$ is key dimensionality.

Scaling by $1 / \sqrt{d_k}$ prevents the dot products from growing excessively large for large projection dimensions, which would otherwise push the softmax function into regions with vanishingly small gradients.
Multi-Head Attention projects $Q, K, V$ into $h$ distinct representation subspaces, enabling the model to attend to information from different representation positions simultaneously.