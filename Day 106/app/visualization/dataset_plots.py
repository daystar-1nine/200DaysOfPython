"""
Dataset visualization plots for Day 106: RNNs & Sequential Text Learning.
Generates Charts 1 to 4.
"""

from pathlib import Path
from typing import Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_dataset_distributions(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Generate Charts 1 to 4 and save as PNG images."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Ham vs Spam distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="label", data=df, palette=["#3498db", "#e74c3c"])
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        pct = count / total * 100
        ax.annotate(f"{count} ({pct:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    plt.title("Chart 1: Class Distribution (Ham vs. Spam)", fontsize=14, fontweight="bold")
    plt.xlabel("Message Category")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig(out / "1_class_distribution.png", dpi=150)
    plt.close()

    # Calculate length metrics
    char_lens = df["text"].apply(len)
    word_lens = df["text"].apply(lambda t: len(t.split()))

    # 2. Message length distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(char_lens, bins=35, kde=True, color="#2ecc71")
    plt.title("Chart 2: Message Character Length Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.axvline(char_lens.median(), color="red", linestyle="--", label=f"Median ({int(char_lens.median())})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "2_message_length_distribution.png", dpi=150)
    plt.close()

    # 3. Token count distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(word_lens, bins=30, kde=True, color="#9b59b6")
    plt.title("Chart 3: Message Word/Token Count Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.axvline(40, color="orange", linestyle="--", label="Target Seq Len (T=40)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "3_token_count_distribution.png", dpi=150)
    plt.close()

    # 4. Character count by class
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="label", y=char_lens, data=df, palette=["#3498db", "#e74c3c"])
    plt.title("Chart 4: Character Count Distribution by Class", fontsize=14, fontweight="bold")
    plt.xlabel("Message Category")
    plt.ylabel("Character Length")
    plt.tight_layout()
    plt.savefig(out / "4_character_count_distribution.png", dpi=150)
    plt.close()
