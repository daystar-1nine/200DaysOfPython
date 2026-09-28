import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List

def plot_single_message_attention(
    tokens: List[str],
    weights: np.ndarray,
    title: str,
    save_path: Path
):
    """
    Renders an attention heatmap for a single message where each token
    is paired with its corresponding attention weight.
    """
    valid_len = min(len(tokens), len(weights))
    valid_tokens = tokens[:valid_len]
    valid_weights = weights[:valid_len]
    
    # Normalize for display
    if valid_weights.sum() > 0:
        valid_weights = valid_weights / valid_weights.sum()
        
    plt.figure(figsize=(max(6, len(valid_tokens) * 0.8), 2.5))
    heatmap_data = np.expand_dims(valid_weights, axis=0)
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar=False,
        xticklabels=valid_tokens,
        yticklabels=["Attention"]
    )
    plt.title(title, fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def generate_attention_example_heatmaps(
    texts: List[str],
    tokenized_texts: List[List[str]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    attention_weights: np.ndarray,
    output_dir: Path
):
    """
    Saves individual attention heatmaps into output/attention_examples/:
      - At least 10 correctly classified spam
      - At least 10 correctly classified ham
      - Misclassified examples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    
    correct_spam = np.where((y_true == 1) & (y_pred == 1))[0]
    correct_ham = np.where((y_true == 0) & (y_pred == 0))[0]
    misclassified = np.where(y_true != y_pred)[0]
    
    # Save up to 10 spam
    for rank, idx in enumerate(correct_spam[:10], start=1):
        plot_single_message_attention(
            tokens=tokenized_texts[idx],
            weights=attention_weights[idx],
            title=f"Spam Correct #{rank} (Prob={y_prob[idx]:.3f})",
            save_path=output_dir / f"correct_spam_{rank:02d}.png"
        )
        
    # Save up to 10 ham
    for rank, idx in enumerate(correct_ham[:10], start=1):
        plot_single_message_attention(
            tokens=tokenized_texts[idx],
            weights=attention_weights[idx],
            title=f"Ham Correct #{rank} (Prob={y_prob[idx]:.3f})",
            save_path=output_dir / f"correct_ham_{rank:02d}.png"
        )
        
    # Save misclassified
    for rank, idx in enumerate(misclassified[:10], start=1):
        actual = "Spam" if y_true[idx] == 1 else "Ham"
        pred = "Spam" if y_pred[idx] == 1 else "Ham"
        plot_single_message_attention(
            tokens=tokenized_texts[idx],
            weights=attention_weights[idx],
            title=f"Error #{rank} (Actual={actual}, Pred={pred}, Prob={y_prob[idx]:.3f})",
            save_path=output_dir / f"error_{rank:02d}.png"
        )
