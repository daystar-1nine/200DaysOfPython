import pandas as pd
import numpy as np

def extract_coefficients(model, feature_names=None) -> pd.DataFrame:
    try:
        if hasattr(model, 'pipeline'):
            regressor = model.pipeline.named_steps['regressor']
            coefs = regressor.coef_
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
        else:
            return pd.DataFrame()
            
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(coefs))]
            
        df = pd.DataFrame({
            "Feature": feature_names[:len(coefs)],
            "Coefficient": coefs
        })
        df["Abs_Coefficient"] = df["Coefficient"].abs()
        return df.sort_values(by="Abs_Coefficient", ascending=False)
    except Exception:
        return pd.DataFrame()

def compare_coefficients(models_dict: dict, feature_names=None) -> pd.DataFrame:
    dfs = []
    for name, model in models_dict.items():
        df = extract_coefficients(model, feature_names)
        if not df.empty:
            df = df.rename(columns={"Coefficient": f"{name}_Coef", "Abs_Coefficient": f"{name}_Abs"})
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame()
        
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, on="Feature", how="outer")
        
    return result
