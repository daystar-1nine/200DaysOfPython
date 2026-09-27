import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def plot_single_history(history: Dict, model_name: str, metric: str, output_path: Path):
    """
    Plots training and validation history for a specific metric.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(7, 4.5))
    train_key = f"train_{metric}"
    val_key = f"val_{metric}"
    
    epochs = range(1, len(history[train_key]) + 1)
    plt.plot(epochs, history[train_key], label=f"Train {metric.capitalize()}", color="#1f77b4", lw=2)
    plt.plot(epochs, history[val_key], label=f"Val {metric.capitalize()}", color="#ff7f0e", lw=2, linestyle="--")
    
    plt.title(f"{model_name} - {metric.capitalize()} Progression", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel(metric.capitalize(), fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def generate_training_plots(histories: Dict[str, Dict], charts_dir: Path):
    """
    Generates training loss and accuracy plots for RNN, LSTM, and GRU:
      Plot 5: RNN loss
      Plot 6: LSTM loss
      Plot 7: GRU loss
      Plot 8: RNN accuracy
      Plot 9: LSTM accuracy
      Plot 10: GRU accuracy
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in ["RNN", "LSTM", "GRU"]:
        if model_name in histories:
            hist = histories[model_name]
            # Loss plot
            plot_single_history(
                hist, model_name, "loss",
                charts_dir / f"{model_name.lower()}_loss.png"
            )
            # Accuracy plot
            plot_single_history(
                hist, model_name, "acc",
                charts_dir / f"{model_name.lower()}_accuracy.png"
            )
