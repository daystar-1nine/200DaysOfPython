"""
Parameter calculation and memory footprint estimators for LLM architectures.
Supports breakdown across token embeddings, positional encodings, attention, MLP, and LM head.
"""
from typing import Dict, Any, Union


PRECISION_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp4": 0.5
}


def estimate_parameters(
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    context_length: int = 1024,
    ff_multiplier: int = 4,
    tie_weights: bool = True,
    include_biases: bool = True
) -> Dict[str, Any]:
    """
    Computes exact parameter counts for a decoder-only Transformer (GPT style).
    Returns breakdown across embeddings, transformer blocks, layer norms, and heads.
    """
    d = embed_dim
    d_ff = d * ff_multiplier

    # 1. Embeddings
    token_embed_params = vocab_size * d
    pos_embed_params = context_length * d
    embed_params = token_embed_params + pos_embed_params

    # 2. Transformer Block components
    # LayerNorm 1 (weight + bias)
    ln1_params = 2 * d if include_biases else d

    # Multi-Head Attention: Q, K, V projections + Output projection
    # c_attn: d -> 3d (weights: 3 * d * d, biases: 3d)
    # c_proj: d -> d  (weights: d * d, biases: d)
    qkv_weights = 3 * d * d
    qkv_biases = 3 * d if include_biases else 0
    proj_weights = d * d
    proj_biases = d if include_biases else 0
    attn_params_per_block = qkv_weights + qkv_biases + proj_weights + proj_biases

    # LayerNorm 2 (weight + bias)
    ln2_params = 2 * d if include_biases else d

    # MLP / Feed-Forward Network: c_fc (d -> d_ff) + c_proj (d_ff -> d)
    fc_weights = d * d_ff
    fc_biases = d_ff if include_biases else 0
    mlp_proj_weights = d_ff * d
    mlp_proj_biases = d if include_biases else 0
    mlp_params_per_block = fc_weights + fc_biases + mlp_proj_weights + mlp_proj_biases

    block_params_single = ln1_params + attn_params_per_block + ln2_params + mlp_params_per_block
    all_blocks_params = block_params_single * num_layers

    # 3. Final LayerNorm
    final_ln_params = 2 * d if include_biases else d

    # 4. LM Head (Unembedding)
    if tie_weights:
        lm_head_params = 0  # Reuses token_embed_params
    else:
        lm_head_params = d * vocab_size

    total_params = embed_params + all_blocks_params + final_ln_params + lm_head_params

    return {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "context_length": context_length,
        "ff_multiplier": ff_multiplier,
        "tie_weights": tie_weights,
        "token_embeddings": token_embed_params,
        "position_embeddings": pos_embed_params,
        "single_block": block_params_single,
        "all_blocks": all_blocks_params,
        "attn_per_block": attn_params_per_block,
        "mlp_per_block": mlp_params_per_block,
        "final_layernorm": final_ln_params,
        "lm_head": lm_head_params,
        "total_parameters": total_params,
        "non_embedding_parameters": all_blocks_params + final_ln_params
    }


def parameter_memory(parameters: int, precision: str = "fp32") -> Dict[str, float]:
    """
    Computes static memory footprint required to store model parameters in given precision.
    """
    prec_lower = precision.lower()
    bytes_per_param = PRECISION_BYTES.get(prec_lower, 4.0)
    total_bytes = parameters * bytes_per_param
    total_kb = total_bytes / 1024
    total_mb = total_kb / 1024
    total_gb = total_mb / 1024

    return {
        "parameters": parameters,
        "precision": prec_lower,
        "bytes_per_parameter": bytes_per_param,
        "bytes": total_bytes,
        "kilobytes": round(total_kb, 4),
        "megabytes": round(total_mb, 4),
        "gigabytes": round(total_gb, 6)
    }


def estimate_training_memory(
    parameters: int,
    optimizer: str = "adamw",
    precision: str = "fp32",
    batch_size: int = 32,
    context_length: int = 512,
    num_layers: int = 12,
    embed_dim: int = 768,
    num_heads: int = 12
) -> Dict[str, Any]:
    """
    Estimates total VRAM required during training including:
    - Model parameters
    - Gradients
    - Optimizer states (e.g. AdamW tracks momentum and variance in FP32)
    - Activations (forward pass activations cached for backprop)
    """
    prec_lower = precision.lower()
    bytes_p = PRECISION_BYTES.get(prec_lower, 4.0)

    # 1. Weights memory
    model_bytes = parameters * bytes_p

    # 2. Gradients memory (usually matches model precision)
    grad_bytes = parameters * bytes_p

    # 3. Optimizer state memory
    # Adam / AdamW maintains 1st moment (momentum) and 2nd moment (variance) in FP32 (4 bytes each)
    # In mixed precision (AMP), master weights are also kept in FP32 (4 bytes).
    if optimizer.lower() in ("adam", "adamw"):
        if prec_lower in ("fp16", "bf16"):
            # Master weights (4 bytes) + momentum (4 bytes) + variance (4 bytes) = 12 bytes/param
            opt_bytes = parameters * 12
        else:
            # Momentum (4 bytes) + variance (4 bytes) = 8 bytes/param
            opt_bytes = parameters * 8
    elif optimizer.lower() == "sgd":
        opt_bytes = parameters * 4  # Momentum buffer
    else:
        opt_bytes = parameters * 8

    # 4. Activation memory approximation (per-token activation cache for Transformer block)
    # Standard formula: ~ B * T * num_layers * embed_dim * (34 + 5 * num_heads * context_length / embed_dim)
    # Or simplified empirical estimate: ~ (34 * bytes_p) per layer per token
    activations_bytes = batch_size * context_length * num_layers * embed_dim * 34 * bytes_p

    total_bytes = model_bytes + grad_bytes + opt_bytes + activations_bytes

    return {
        "model_memory_gb": round(model_bytes / (1024 ** 3), 4),
        "gradients_memory_gb": round(grad_bytes / (1024 ** 3), 4),
        "optimizer_memory_gb": round(opt_bytes / (1024 ** 3), 4),
        "activations_memory_gb": round(activations_bytes / (1024 ** 3), 4),
        "total_training_memory_gb": round(total_bytes / (1024 ** 3), 4),
        "total_training_memory_mb": round(total_bytes / (1024 ** 2), 2)
    }


def format_bytes(num_bytes: Union[int, float]) -> str:
    """Formats raw bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"
