"""
Self-attention visualization for BERT transformer layers.
"""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from transformers import BertTokenizer


def visualize_bert_attention(
    text: str,
    model: nn.Module,
    tokenizer: BertTokenizer,
    output_path: Path,
    layer_idx: int = -1,
    title: Optional[str] = None
) -> None:
    """
    Extracts multi-head attention weights from BERT and plots a token-by-token attention heatmap.

    Args:
        text: Input string sequence.
        model: Trained or pretrained BertSpamClassifier.
        tokenizer: BertTokenizer.
        output_path: File path to save the heatmap image.
        layer_idx: Which layer's attention to plot (default: -1 for final layer).
        title: Optional plot title.
    """
    model.eval()
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        _, _, attentions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    if attentions is None:
        raise ValueError("Model did not return attentions. Ensure output_attentions=True is enabled.")

    # attentions is a tuple of length num_layers, each (batch_size, num_heads, seq_len, seq_len)
    layer_attn = attentions[layer_idx][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
    num_heads = layer_attn.shape[0]

    # Average attention across all attention heads
    avg_attn = layer_attn.mean(axis=0)  # (seq_len, seq_len)

    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.45), max(5, len(tokens) * 0.4)))
    sns.heatmap(
        avg_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        annot=len(tokens) <= 16,
        fmt=".2f",
        cbar_kws={"label": "Attention Weight"},
        ax=ax
    )

    resolved_title = title or f"BERT Self-Attention (Layer {layer_idx if layer_idx >= 0 else len(attentions) + layer_idx + 1})"
    ax.set_title(resolved_title, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Key Tokens", fontsize=10)
    ax.set_ylabel("Query Tokens", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Attention heatmap saved to: {output_path}")
