"""
Training dynamics visualization for Day 106: RNNs & Sequential Text Learning.
Generates Charts 5 to 8.
"""

from pathlib import Path
from typing import Dict, List, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history: Dict[str, List[float]], output_dir: Union[str, Path]) -> None:
    """Generate Charts 5 to 8 and save as PNG images."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 5. Training Loss
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], marker="o", color="#e74c3c", linewidth=2.2, label="Train Loss")
    plt.title("Chart 5: SimpleRNN Training Loss Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "5_training_loss.png", dpi=150)
    plt.close()

    # 6. Validation Loss
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_loss"], marker="s", color="#e67e22", linewidth=2.2, label="Validation Loss")
    plt.title("Chart 6: SimpleRNN Validation Loss Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "6_validation_loss.png", dpi=150)
    plt.close()

    # 7. Training Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], marker="o", color="#2ecc71", linewidth=2.2, label="Train Accuracy")
    plt.title("Chart 7: SimpleRNN Training Accuracy Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Score")
    plt.ylim(0.70, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "7_training_accuracy.png", dpi=150)
    plt.close()

    # 8. Validation Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_acc"], marker="^", color="#3498db", linewidth=2.2, label="Validation Accuracy")
    plt.title("Chart 8: SimpleRNN Validation Accuracy Across Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Score")
    plt.ylim(0.70, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "8_validation_accuracy.png", dpi=150)
    plt.close()
