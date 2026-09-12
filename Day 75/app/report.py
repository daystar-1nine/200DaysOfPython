import os
import pandas as pd

def generate_report(comparison_df, best_model_name, tuning_results, insights, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("Regression Analysis Report\n")
        f.write("==========================\n\n")
        
        f.write(f"Best Model: {best_model_name}\n\n")
        
        f.write("Model Comparison:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("Hyperparameter Tuning:\n")
        f.write(f"Poly best degree: {tuning_results['polynomial']['best_degree']}\n")
        f.write(f"Ridge best alpha: {tuning_results['ridge']['best_alpha']}\n")
        f.write(f"Lasso best alpha: {tuning_results['lasso']['best_alpha']}\n\n")
        
        f.write("Key Insights:\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
