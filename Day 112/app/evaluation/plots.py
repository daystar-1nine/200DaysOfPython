"""
Plotting and automated experiment report generation for MiniGPT.
"""
from pathlib import Path
from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from app.tokenizer.char_tokenizer import CharacterTokenizer
from app.generation.sampling import generate_with_strategy
from app.evaluation.metrics import evaluate_text_diversity
from app.evaluation.repetition import repetition_rate


def plot_training_history(history: Dict[str, List[Any]], output_path: Path) -> None:
    """Plots training and validation loss curves and perplexity."""
    steps = history["step"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    ppl = history["perplexity"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curves
    ax1.plot(steps, train_loss, "o-", color="#2980b9", label="Train Loss", linewidth=2)
    ax1.plot(steps, val_loss, "s--", color="#e74c3c", label="Validation Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss (Teacher Forcing)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Optimization Step", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    # Perplexity Curve
    ax2.plot(steps, ppl, "^-", color="#27ae60", label="Validation Perplexity", linewidth=2)
    ax2.set_title("Validation Perplexity (PPL = exp(Loss))", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimization Step", fontsize=10)
    ax2.set_ylabel("Perplexity", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {output_path}")


def run_temperature_experiment(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str = "First Citizen:\n",
    temperatures: tuple = (0.3, 0.7, 1.0, 1.3),
    output_md: Path = None
) -> pd.DataFrame:
    """
    Generates text across different temperatures, evaluates repetition and diversity,
    and outputs a comprehensive markdown report.
    """
    records = []
    markdown_lines = [
        "# MiniGPT: Temperature Sensitivity Analysis\n",
        f"**Prompt:** `{prompt.strip()}`\n\n",
        "Temperature modifies logit sharpness: $z' = z / T$.\n\n",
        "| Temperature | Observation | Repetition (3-gram) | Distinct-1 | Distinct-2 |\n",
        "|---|---|:---:|:---:|:---:|\n"
    ]

    for temp in temperatures:
        gen = generate_with_strategy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=150,
            temperature=temp
        )
        words = gen.strip().split()
        rep = repetition_rate(words, n=3) if len(words) >= 3 else 0.0
        div = evaluate_text_diversity(gen)

        obs = "Sharp, deterministic, repetitive" if temp < 0.5 else \
              "Balanced, coherent continuation" if temp <= 1.0 else \
              "High entropy, diverse, degraded syntax"

        records.append({
            "temperature": temp,
            "repetition_rate": rep,
            "distinct_1": div["distinct_1_word"],
            "distinct_2": div["distinct_2_word"],
            "text": gen
        })

        markdown_lines.append(
            f"| **{temp:.1f}** | {obs} | {rep:.4f} | {div['distinct_1_word']:.4f} | {div['distinct_2_word']:.4f} |\n"
        )

    markdown_lines.append("\n## Generated Text Samples\n\n")
    for r in records:
        markdown_lines.append(f"### Temperature = {r['temperature']}\n```text\n{r['text']}\n```\n\n")

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.writelines(markdown_lines)
        print(f"Temperature comparison report saved to: {output_md}")

    return pd.DataFrame(records)


def run_strategy_comparison(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str = "MENENIUS:\n",
    output_md: Path = None
) -> pd.DataFrame:
    """
    Compares Greedy, Pure Sampling, Top-k, Top-p, and Combined decoding.
    """
    strategies = [
        ("Greedy", 0.0, None, None, "Deterministic argmax; prone to looping repetitions"),
        ("Pure Sampling", 1.0, None, None, "Unconstrained categorical sampling across full vocab"),
        ("Top-k (k=20)", 0.8, 20, None, "Truncates to top 20 candidate tokens; eliminates low-prob tail"),
        ("Top-p (p=0.9)", 0.8, None, 0.9, "Nucleus sampling over dynamic 90% cumulative probability mass"),
        ("Top-k + Top-p", 0.8, 20, 0.9, "Combined top-k and nucleus thresholding for peak coherence")
    ]

    records = []
    markdown_lines = [
        "# MiniGPT: Decoding Strategy Benchmark\n",
        f"**Prompt:** `{prompt.strip()}`\n\n",
        "| Strategy | Temperature | k | p | Repetition (3-gram) | Distinct-1 | Distinct-2 | Qualitative Behavior |\n",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|---|\n"
    ]

    for name, temp, k, p, desc in strategies:
        gen = generate_with_strategy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=150,
            temperature=temp,
            top_k=k,
            top_p=p
        )
        words = gen.strip().split()
        rep = repetition_rate(words, n=3) if len(words) >= 3 else 0.0
        div = evaluate_text_diversity(gen)

        records.append({
            "strategy": name,
            "temperature": temp,
            "top_k": k,
            "top_p": p,
            "repetition_rate": rep,
            "distinct_1": div["distinct_1_word"],
            "distinct_2": div["distinct_2_word"],
            "text": gen
        })

        k_str = str(k) if k is not None else "-"
        p_str = str(p) if p is not None else "-"
        markdown_lines.append(
            f"| **{name}** | {temp:.1f} | {k_str} | {p_str} | {rep:.4f} | {div['distinct_1_word']:.4f} | {div['distinct_2_word']:.4f} | {desc} |\n"
        )

    markdown_lines.append("\n## Generated Samples by Decoding Strategy\n\n")
    for r in records:
        markdown_lines.append(f"### Strategy: {r['strategy']}\n```text\n{r['text']}\n```\n\n")

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.writelines(markdown_lines)
        print(f"Generation strategy comparison saved to: {output_md}")

    return pd.DataFrame(records)
