"""Markdown report generator for Day 103 Word2Vec Engine."""
from pathlib import Path
from typing import Dict, Any
import pandas as pd

def generate_report(
    analysis_data: Dict[str, Any],
    top_neighbors_df: pd.DataFrame,
    losses: list,
    output_path: Path
):
    # Format neighbors table manually
    headers = list(top_neighbors_df.columns)
    table_n = "| " + " | ".join(headers) + " |\n"
    table_n += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in top_neighbors_df.head(15).iterrows():
        table_n += "| " + " | ".join([str(v) for v in row]) + " |\n"

    # Semantic pairs
    pairs_md = ""
    for pair, sim in analysis_data["semantic_pairs"].items():
        pairs_md += f"- **{pair}**: `{sim:.4f}`\n"

    md = f"""# Day 103 — Word Embeddings & Semantic Representation Report

## 1. Model & Architecture Summary
- **Model**: Word2Vec (Skip-Gram with Negative Sampling from scratch using NumPy)
- **Vocabulary Size**: {analysis_data['vocab_size']} words
- **Embedding Dimensions**: {analysis_data['embedding_dim']}
- **Initial Training Loss**: {losses[0]:.4f}
- **Final Training Loss (Epoch {len(losses)})**: {losses[-1]:.4f}
- **Embedding Norm Statistics**: Mean={analysis_data['norm_mean']:.4f}, Std={analysis_data['norm_std']:.4f}, Min={analysis_data['norm_min']:.4f}, Max={analysis_data['norm_max']:.4f}

---

## 2. Semantic Pair Cosine Similarities
{pairs_md}

---

## 3. Nearest Neighbor Samples
{table_n}

---

## 4. Key Engineering Insights
1. **Distributional Learning**: Words sharing similar contexts (e.g. *cat*, *dog*, *kitten*, *puppy*) move closer in dense vector space compared to distant technical terms (*python*, *java*).
2. **Computational Efficiency of Negative Sampling**: By replacing the $O(V)$ full softmax with $1 + K$ binary sigmoid updates, training time per step scales as $O(D \\times K)$ instead of $O(D \\times V)$.
3. **Limitation of Static Embeddings**: Static embeddings assign a single fixed vector per token, making polysemy (e.g. *bank* as river bank vs financial bank) unresolvable without contextual transformers.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Generated report at {output_path}")
