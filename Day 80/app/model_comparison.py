from typing import Dict, Any, List
import pandas as pd

def build_model_comparison_table(
    cv_results: Dict[str, Dict[str, Dict[str, float]]],
    test_results: Dict[str, Dict[str, float]],
    business_costs: Dict[str, float]
) -> pd.DataFrame:
    """Assemble standardized model comparison matrix across CV, test, and business metrics."""
    rows = []
    
    for model_name in test_results.keys():
        cv_m = cv_results.get(model_name, {})
        test_m = test_results.get(model_name, {})
        cost = business_costs.get(model_name, 0.0)
        
        row = {
            'Model': model_name,
            'CV_ROC_AUC_Mean': cv_m.get('roc_auc', {}).get('test_mean', 0.0),
            'CV_ROC_AUC_Std': cv_m.get('roc_auc', {}).get('test_std', 0.0),
            'CV_AP_Mean': cv_m.get('average_precision', {}).get('test_mean', 0.0),
            'CV_F1_Mean': cv_m.get('f1', {}).get('test_mean', 0.0),
            'Test_Accuracy': test_m.get('accuracy', 0.0),
            'Test_Precision': test_m.get('precision', 0.0),
            'Test_Recall': test_m.get('recall', 0.0),
            'Test_F1': test_m.get('f1', 0.0),
            'Test_ROC_AUC': test_m.get('roc_auc', 0.0),
            'Test_AP': test_m.get('average_precision', 0.0),
            'Test_Business_Cost': cost
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    return df.sort_values(by='Test_ROC_AUC', ascending=False).reset_index(drop=True)

def select_best_model(df_comparison: pd.DataFrame, primary_metric: str = 'Test_ROC_AUC') -> str:
    """Select champion model based on primary ranking metric."""
    best_idx = df_comparison[primary_metric].idxmax()
    return str(df_comparison.loc[best_idx, 'Model'])
