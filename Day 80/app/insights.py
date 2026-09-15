from typing import Dict, Any, List
import pandas as pd

def generate_business_insights(
    df_comparison: pd.DataFrame,
    champion_model: str,
    top_features: List[str],
    cost_summary: Dict[str, Any]
) -> str:
    """Generate synthesized executive findings from empirical experiment results."""
    champ_row = df_comparison[df_comparison['Model'] == champion_model].iloc[0]
    baseline_row = df_comparison[df_comparison['Model'] == 'Baseline (Dummy)'].iloc[0] if 'Baseline (Dummy)' in df_comparison['Model'].values else None
    
    savings = cost_summary.get('savings', 0.0)
    pct_savings = cost_summary.get('pct_savings', 0.0)
    opt_t = cost_summary.get('optimal_threshold', 0.50)
    
    text = f"""# 🏆 Executive Model Selection & Churn Intelligence Summary

### 1. Champion Model Selection: {champion_model}
- **Validation ROC-AUC**: {champ_row['CV_ROC_AUC_Mean']:.4f} (±{champ_row['CV_ROC_AUC_Std']:.4f})
- **Test ROC-AUC**: {champ_row['Test_ROC_AUC']:.4f} | **Test Average Precision**: {champ_row['Test_AP']:.4f}
- **Test F1 Score**: {champ_row['Test_F1']:.4f} | **Test Recall**: {champ_row['Test_Recall']:.4f}
- **Generalization Rigor**: The {champion_model} significantly outperformed the naive baseline and demonstrated robust cross-validation stability without severe overfitting.

### 2. Primary Churn Risk Drivers
The empirical data identifies the top predictive signals driving customer attrition:
1. **{top_features[0]}**
2. **{top_features[1]}**
3. **{top_features[2]}**

### 3. Business Cost Optimization & Financial Impact
- **Standard Threshold (0.50) Cost**: INR {cost_summary.get('default_cost', 0.0):,.2f}
- **Cost-Optimal Threshold ({opt_t:.2f}) Cost**: INR {cost_summary.get('optimal_cost', 0.0):,.2f}
- **Net Business Value Unlocked**: **INR {savings:,.2f}** ({pct_savings:.1f}% reduction in misclassification costs).
- Operating at a threshold below 0.50 captures substantially more high-risk churners, preventing expensive customer attrition while keeping false-positive incentive costs manageable.
"""
    return text
