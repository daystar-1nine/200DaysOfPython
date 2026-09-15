from typing import Dict, Any

def generate_executive_insights(
    benchmark_metrics: Dict[str, Dict[str, float]],
    rf_oob_score: float,
    top_mdi_features: list,
    cost_comparison: Dict[str, float]
) -> str:
    """Generate automated narrative executive summary."""
    rf_f1 = benchmark_metrics.get('Random Forest', {}).get('f1_score', 0.0)
    dt_f1 = benchmark_metrics.get('Decision Tree', {}).get('f1_score', 0.0)
    lr_f1 = benchmark_metrics.get('Logistic Regression', {}).get('f1_score', 0.0)
    tuned_rf_f1 = benchmark_metrics.get('Tuned Random Forest', {}).get('f1_score', 0.0)
    
    default_cost = cost_comparison.get('default_cost', 0.0)
    optimal_cost = cost_comparison.get('optimal_cost', 0.0)
    savings = cost_comparison.get('savings', 0.0)
    pct_savings = cost_comparison.get('pct_savings', 0.0)
    opt_t = cost_comparison.get('optimal_threshold', 0.50)
    
    text = f"""# 🏆 Executive Churn Intelligence & Ensemble Analysis

### 1. Model Superiority & Variance Reduction
- **Baseline Logistic Regression**: F1 = {lr_f1:.4f}
- **Single Decision Tree**: F1 = {dt_f1:.4f}
- **Random Forest (100 Trees)**: F1 = {rf_f1:.4f} (OOB Score = {rf_oob_score:.4f})
- **Tuned Random Forest**: F1 = {tuned_rf_f1:.4f}
- The Random Forest ensemble demonstrates significant variance reduction over the single Decision Tree, avoiding brittle decision boundaries and decorrelating individual tree errors through bootstrap aggregation and feature subspace sampling.

### 2. Primary Churn Drivers (Top Predictors)
- The strongest predictive signals identified by Gini impurity reduction (MDI) are:
  1. **{top_mdi_features[0]}**
  2. **{top_mdi_features[1]}**
  3. **{top_mdi_features[2]}**

### 3. Business Cost Optimization
- At the standard threshold of **0.50**, total intervention cost is **INR {default_cost:,.2f}**.
- By shifting to the cost-optimal decision threshold of **{opt_t:.2f}**, total cost drops to **INR {optimal_cost:,.2f}**.
- **Net Business Value Unlocked**: **INR {savings:,.2f}** ({pct_savings:.1f}% reduction in misclassification costs).
"""
    return text
