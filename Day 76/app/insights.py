import pandas as pd
from typing import List

def generate_insights(metrics: dict, cm_dict: dict, threshold_df: pd.DataFrame, optimal_threshold: float, coef_df: pd.DataFrame, cost_analysis: pd.DataFrame) -> List[str]:
    insights = []
    
    # 1-3. Model Performance Insights
    acc = metrics.get('Accuracy', 0)
    insights.append(f"The model achieves an overall accuracy of {acc:.2%}.")
    insights.append(f"Precision is {metrics.get('Precision', 0):.2f} and Recall is {metrics.get('Recall', 0):.2f}.")
    insights.append(f"The F1 score is {metrics.get('F1', 0):.2f}.")
    if 'ROC_AUC' in metrics:
        insights.append(f"The model's ability to distinguish classes is strong, with an ROC AUC of {metrics['ROC_AUC']:.2f}.")
        
    # 4-6. Confusion Matrix Insights
    tp = cm_dict['TP']
    tn = cm_dict['TN']
    fp = cm_dict['FP']
    fn = cm_dict['FN']
    total = tp + tn + fp + fn
    insights.append(f"Out of {total} cases, the model correctly identified {tp} true churners.")
    insights.append(f"The model missed {fn} actual churners (False Negatives).")
    insights.append(f"The model incorrectly predicted churn for {fp} retained customers (False Positives).")
    
    # 7-9. Feature Importance Insights
    coef_df = coef_df.sort_values(by='Odds_Ratio', ascending=False)
    top_features = coef_df.head(3)['Feature'].tolist()
    bottom_features = coef_df.tail(3)['Feature'].tolist()
    insights.append(f"The features most strongly associated with an increased likelihood of churn are {', '.join(top_features)}.")
    insights.append(f"Features strongly associated with retention include {', '.join(bottom_features)}.")
    
    # 10-12. Threshold & Cost Insights
    insights.append(f"The mathematically optimal threshold based on F1 score is {optimal_threshold:.2f}.")
    
    min_cost_row = cost_analysis.loc[cost_analysis['Total_Cost'].idxmin()]
    min_cost_threshold = min_cost_row['Threshold']
    min_cost_value = min_cost_row['Total_Cost']
    insights.append(f"When optimizing for business cost, the ideal threshold is {min_cost_threshold:.2f}.")
    insights.append(f"At this cost-optimal threshold, the estimated business cost is ${min_cost_value:,.2f}.")
    
    # 13-15. General Business Strategy Insights
    insights.append("Consider targeting retention offers to customers identified by the model to balance cost and benefit.")
    insights.append("Focus on improving services related to the highest risk features identified.")
    
    return insights
