import os
import pandas as pd

try:
    from app.config import AppConfig
except ImportError:
    from config import AppConfig

def save_report(metrics: dict, coef_df: pd.DataFrame, vif_df: pd.DataFrame, residual_stats: dict, insights: list[str], config: AppConfig):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save CSVs
    coef_df.to_csv(config.COEFFICIENTS_CSV, index=False)
    vif_df.to_csv(config.VIF_REPORT_CSV, index=False)
    
    # Text report
    with open(config.REGRESSION_REPORT_TXT, 'w') as f:
        f.write("="*50 + "\n")
        f.write("MULTIPLE LINEAR REGRESSION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("METRICS\n")
        f.write("-" * 20 + "\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
        
        f.write("\nINSIGHTS\n")
        f.write("-" * 20 + "\n")
        for i, ins in enumerate(insights, 1):
            f.write(f"{i}. {ins}\n")
            
        f.write("\nRESIDUALS\n")
        f.write("-" * 20 + "\n")
        for k, v in residual_stats.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
