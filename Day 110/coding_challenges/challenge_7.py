"""
Challenge 7: Implement count_transformer_parameters() and compare with framework.
"""
import torch
import torch.nn as nn


def count_transformer_parameters_analytical(
    vocab_size: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    head_dim_hidden: int = 64
) -> int:
    """
    Calculates exact parameter count analytically for a Transformer classifier.
    """
    # 1. Token Embedding: vocab_size * d_model (sinusoidal PE has 0 trainable params)
    embed_params = vocab_size * d_model

    # 2. Per Transformer Block:
    # MHA: Q, K, V, O projections (weight: d_model * d_model, bias: d_model)
    mha_params = 4 * (d_model * d_model + d_model)

    # LayerNorm 1: weight (d_model) + bias (d_model)
    ln1_params = 2 * d_model

    # FFN: linear1 (d_model * d_ff + d_ff) + linear2 (d_ff * d_model + d_model)
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)

    # LayerNorm 2: weight (d_model) + bias (d_model)
    ln2_params = 2 * d_model

    block_params = mha_params + ln1_params + ffn_params + ln2_params
    total_encoder_params = num_layers * block_params

    # 3. Classifier head: Linear(d_model, 64) + Linear(64, 1)
    head_params = (d_model * head_dim_hidden + head_dim_hidden) + (head_dim_hidden * 1 + 1)

    return embed_params + total_encoder_params + head_params


if __name__ == "__main__":
    import sys
    from pathlib import Path
    day_dir = Path(__file__).resolve().parent.parent
    if str(day_dir) not in sys.path:
        sys.path.insert(0, str(day_dir))

    from app.models.classifier import MiniTransformerClassifier

    vocab_size = 100
    d_model = 64
    d_ff = 128
    num_layers = 2

    # Analytical count
    analytical = count_transformer_parameters_analytical(vocab_size, d_model, d_ff, num_layers)

    # Empirical framework count
    model = MiniTransformerClassifier(
        vocab_size=vocab_size,
        max_length=20,
        d_model=d_model,
        num_heads=4,
        d_ff=d_ff,
        num_layers=num_layers
    )
    empirical = model.count_parameters()

    print(f"Analytical Parameters: {analytical}")
    print(f"Empirical Parameters:  {empirical}")
    assert analytical == empirical, f"Mismatch: {analytical} != {empirical}"
    print("Challenge 7 passed!")
