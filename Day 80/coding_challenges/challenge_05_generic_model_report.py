"""Challenge 5: Generic Model Selection Report Generator."""
from typing import Dict, Any, List

def generate_model_report(
    best_model_name: str,
    cv_score: float,
    test_score: float,
    best_params: Dict[str, Any],
    threshold: float,
    metrics: Dict[str, float],
    business_cost: float,
    top_features: List[str]
) -> str:
    """Generate professional executive summary text."""
    report = f"""==================================================
MODEL SELECTION & EVALUATION REPORT
==================================================
Selected Model:       {best_model_name}
Cross-Validation ROC: {cv_score:.4f}
Final Test ROC-AUC:   {test_score:.4f}
Best Parameters:      {best_params}

Operating Decision Point:
- Operating Threshold:{threshold:.2f}
- Precision:          {metrics.get('precision', 0.0):.4f}
- Recall:             {metrics.get('recall', 0.0):.4f}
- F1-Score:           {metrics.get('f1', 0.0):.4f}
- Business Loss:      INR {business_cost:,.2f}

Top Predictive Features:
"""
    for idx, feat in enumerate(top_features[:5], 1):
        report += f"  {idx}. {feat}\n"
    report += "=================================================="
    return report

if __name__ == '__main__':
    r = generate_model_report(
        best_model_name='Random Forest',
        cv_score=0.9754,
        test_score=0.9825,
        best_params={'max_depth': 12, 'n_estimators': 50},
        threshold=0.30,
        metrics={'precision': 0.9577, 'recall': 0.9174, 'f1': 0.9370},
        business_cost=12000.0,
        top_features=['Contract_Type', 'Tenure_Months', 'Monthly_Charges']
    )
    print(r)
    assert 'Random Forest' in r and 'INR 12,000.00' in r
    print("Challenge 5 passed!")
