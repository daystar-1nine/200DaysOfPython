
import sys
import logging
import pandas as pd
from app.loader import load_data
from app.cleaner import clean_data
from app.validator import validate_data
from app.pipeline import prepare_data
from app.models import (train_baseline, train_logistic_regression, train_decision_tree, 
                       train_random_forest, train_gradient_boosting, train_tuned_gradient_boosting)
from app.evaluation import evaluate_model
from app.business_cost import find_optimal_threshold
from app.visualization import (plot_model_comparison, plot_roc_curves, plot_pr_curves, 
                              plot_confusion_matrices, plot_feature_importance, 
                              plot_learning_rate_sensitivity, plot_n_estimators_sensitivity,
                              plot_tree_depth_sensitivity, plot_business_cost_curve)

# Fix stdout encoding for powershell
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Starting Day 81 Gradient Boosting Engine...")
    
    # 1. Load Data
    df_raw = load_data()
    
    # 2. Clean & Validate
    df_clean = clean_data(df_raw)
    validate_data(df_clean)
    
    # 3. Preprocess
    X_train, X_test, y_train, y_test, feature_names, _ = prepare_data(df_clean)
    print(f"Data prepared. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 4. Train Models
    models = {}
    print("Training Baseline...")
    models['Baseline'] = train_baseline(X_train, y_train)
    
    print("Training Logistic Regression...")
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    
    print("Training Decision Tree...")
    models['Decision Tree'] = train_decision_tree(X_train, y_train)
    
    print("Training Random Forest...")
    models['Random Forest'] = train_random_forest(X_train, y_train)
    
    print("Training Gradient Boosting...")
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    
    print("Training Tuned Gradient Boosting (GridSearch)...")
    models['Tuned Gradient Boosting'] = train_tuned_gradient_boosting(X_train, y_train)
    
    # 5. Evaluate
    results = []
    for name, model in models.items():
        res = evaluate_model(model, X_test, y_test, name)
        results.append(res)
        
    results_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    
    # 6. Business Cost Optimization (Using Tuned GB)
    best_model = models['Tuned Gradient Boosting']
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fp_cost = 50  # Cost of false positive (e.g., offering discount to non-churner)
    fn_cost = 200 # Cost of false negative (e.g., losing a churning customer)
    
    opt_t, opt_c, thresholds, costs = find_optimal_threshold(y_test, y_prob, fp_cost, fn_cost)
    print(f"\nOptimal Threshold for Business Cost: {opt_t:.3f}")
    print(f"Estimated Minimal Cost: ${opt_c:.2f}")
    
    # 7. Visualizations
    print("\nGenerating Visualizations (15+ charts)...")
    plot_model_comparison(results_df) # 5 charts
    plot_roc_curves(models, X_test, y_test) # 1 chart
    plot_pr_curves(models, X_test, y_test) # 1 chart
    plot_confusion_matrices(models, X_test, y_test) # 6 charts
    plot_feature_importance(models['Random Forest'], feature_names, model_name='Random Forest') # 1 chart
    plot_feature_importance(best_model, feature_names, model_name='Tuned Gradient Boosting') # 1 chart
    
    print("Generating Sensitivity Analysis...")
    plot_learning_rate_sensitivity(X_train, y_train, X_test, y_test) # 1 chart
    plot_n_estimators_sensitivity(X_train, y_train, X_test, y_test) # 1 chart
    plot_tree_depth_sensitivity(X_train, y_train, X_test, y_test) # 1 chart
    plot_business_cost_curve(thresholds, costs, opt_t, opt_c) # 1 chart
    
    print("\n✅ Day 81 Pipeline Complete! All outputs saved to Day 81/output/charts/")

if __name__ == "__main__":
    main()
