
import sys
import pandas as pd
from app.config import Config
from app.data.loader import load_data
from app.data.cleaner import clean_data
from app.data.validator import validate_data
from app.preprocessing.pipeline import prepare_data
from app.models import get_models
from app.evaluation.metrics import evaluate_model
from app.evaluation.cross_validation import evaluate_cv
from app.visualizations import generate_all_charts

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("🚀 Starting Day 83 Modern Boosting Benchmark...")
    df = load_data()
    df = clean_data(df)
    validate_data(df)
    
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    
    models = get_models()
    
    results = []
    cv_results = []
    
    for name, model in models.items():
        print(f"Training and Evaluating {name}...")
        res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results.append(res)
        
        cv_res = evaluate_cv(model, X_train, y_train, name)
        cv_results.append(cv_res)
        
    df_results = pd.DataFrame(results)
    df_cv = pd.DataFrame(cv_results)
    
    # Merge for final table
    final_benchmark = pd.merge(df_cv, df_results, on='Model')
    
    print("\n--- Modern Boosting Benchmark ---")
    print(final_benchmark[['Model', 'CV ROC-AUC', 'Test ROC-AUC', 'Average Precision', 'F1 Score', 'Train Time']].to_string(index=False))
    
    print("\nGenerating Visualizations (20+ charts)...")
    generate_all_charts(final_benchmark, models, X_test, y_test, feature_names)
    
    print("✅ Benchmark complete.")

if __name__ == "__main__":
    main()
