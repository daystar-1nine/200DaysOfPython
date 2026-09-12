"""
Day 72 — Report and Export Engine
Exports matrices to CSV and formats comprehensive executive ASCII analysis report.
"""
import os
from typing import List, Dict, Any
import pandas as pd

def export_csv_results(cov_mat: pd.DataFrame, corr_mat: pd.DataFrame, pairwise_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cov_mat.to_csv(os.path.join(output_dir, "covariance_matrix.csv"))
    corr_mat.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    pairwise_df.to_csv(os.path.join(output_dir, "correlation_results.csv"), index=False)

def format_ascii_report(
    n_rows: int,
    n_cols: int,
    top_pos: pd.DataFrame,
    top_neg: pd.DataFrame,
    mc_df: pd.DataFrame,
    divergent_df: pd.DataFrame,
    insights: List[str],
    q_and_a: List[Dict[str, str]]
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("        BUSINESS RELATIONSHIP & CORRELATION ANALYSIS REPORT        ")
    lines.append("=" * 70)
    lines.append(f"Analyzed Rows:                {n_rows:,}")
    lines.append(f"Analyzed Numerical Features:  {n_cols}")
    lines.append(f"Total Pairwise Combinations:  {n_cols * (n_cols - 1) // 2}")
    lines.append("-" * 70)
    
    lines.append("\n" + "=" * 70)
    lines.append("1. TOP POSITIVE RELATIONSHIPS")
    lines.append("=" * 70)
    lines.append(f"{'Variable Pair':<30} | {'Pearson r':<10} | {'Spearman rho':<12} | {'95% CI':<15}")
    lines.append("-" * 70)
    for _, r in top_pos.iterrows():
        pair_name = f"{r['Variable_1']} - {r['Variable_2']}"
        ci_str = f"[{r['CI_Lower']:.2f}, {r['CI_Upper']:.2f}]"
        lines.append(f"{pair_name:<30} | {r['Pearson_r']:<10.3f} | {r['Spearman_rho']:<12.3f} | {ci_str:<15}")
        
    lines.append("\n" + "=" * 70)
    lines.append("2. TOP INVERSE (NEGATIVE) RELATIONSHIPS")
    lines.append("=" * 70)
    lines.append(f"{'Variable Pair':<30} | {'Pearson r':<10} | {'Spearman rho':<12} | {'95% CI':<15}")
    lines.append("-" * 70)
    for _, r in top_neg.iterrows():
        pair_name = f"{r['Variable_1']} - {r['Variable_2']}"
        ci_str = f"[{r['CI_Lower']:.2f}, {r['CI_Upper']:.2f}]"
        lines.append(f"{pair_name:<30} | {r['Pearson_r']:<10.3f} | {r['Spearman_rho']:<12.3f} | {ci_str:<15}")
        
    lines.append("\n" + "=" * 70)
    lines.append("3. POTENTIAL MULTICOLLINEARITY (Threshold |r| >= 0.75)")
    lines.append("=" * 70)
    if mc_df.empty:
        lines.append("No pairwise multicollinearity exceeding threshold detected.")
    else:
        for _, r in mc_df.iterrows():
            lines.append(f"  [!] {r['Variable_1']} <--> {r['Variable_2']} : Pearson r = {r['Pearson_r']:.3f}")
            lines.append("      Advisory: Highly redundant predictors; consider dropping one or dimensionality reduction.")
            
    lines.append("\n" + "=" * 70)
    lines.append("4. NON-LINEAR / MONOTONIC DIVERGENCE (|r - rho| >= 0.20)")
    lines.append("=" * 70)
    if divergent_df.empty:
        lines.append("No extreme divergence between Pearson and Spearman identified.")
    else:
        for _, r in divergent_df.iterrows():
            lines.append(f"  [*] {r['Variable_1']} <--> {r['Variable_2']} : Pearson r = {r['Pearson_r']:.3f}, Spearman rho = {r['Spearman_rho']:.3f} (Diff = {r['Abs_Diff']:.3f})")
            lines.append("      Diagnostic: Likely non-linear curvature or presence of influential leverage points.")
            
    lines.append("\n" + "=" * 70)
    lines.append("5. 10 CORE BUSINESS QUESTIONS ANSWERED")
    lines.append("=" * 70)
    for item in q_and_a:
        lines.append(f"\n{item['Question']}")
        lines.append(f"  Answer: {item['Answer']}")
        
    lines.append("\n" + "=" * 70)
    lines.append("6. AUTOMATED STRATEGIC INSIGHTS & CAUSAL GOVERNANCE")
    lines.append("=" * 70)
    for idx, insight in enumerate(insights, 1):
        lines.append(f"{idx}. {insight}\n")
        
    lines.append("=" * 70)
    lines.append("                      END OF ANALYSIS REPORT                       ")
    lines.append("=" * 70)
    
    return "\n".join(lines)
