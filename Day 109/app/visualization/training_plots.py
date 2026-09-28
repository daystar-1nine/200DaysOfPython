import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def plot_history(history: Dict, model_name: str, metric: str, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(6.5, 4.2))
    train_key = f"train_{metric}"
    val_key = f"val_{metric}"
    
    epochs = range(1, len(history[train_key]) + 1)
    plt.plot(epochs, history[train_key], label=f"Train {metric.capitalize()}", color="#1f77b4", lw=2)
    plt.plot(epochs, history[val_key], label=f"Val {metric.capitalize()}", color="#ff7f0e", lw=2, linestyle="--")
    
    plt.title(f"{model_name} - {metric.capitalize()}", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def generate_training_plots(histories: Dict[str, Dict], charts_dir: Path):
    """
    Plots 4-7:
      Plot 4: GRU loss
      Plot 5: GRU + Attention loss
      Plot 6: GRU accuracy
      Plot 7: GRU + Attention accuracy
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    if "GRU" in histories:
        plot_history(histories["GRU"], "GRU Baseline", "loss", charts_dir / "04_gru_loss.png")
        plot_history(histories["GRU"], "GRU Baseline", "acc", charts_dir / "06_gru_accuracy.png")
        
    if "GRU + Attention" in histories:
        plot_history(histories["GRU + Attention"], "GRU + Attention", "loss", charts_dir / "05_gru_attention_loss.png")
        plot_history(histories["GRU + Attention"], "GRU + Attention", "acc", charts_dir / "07_gru_attention_accuracy.png")
