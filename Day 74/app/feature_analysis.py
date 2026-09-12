import pandas as pd

def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    return df[numeric_cols].corr()

def top_correlations(corr_df: pd.DataFrame, target: str, n: int = 10) -> pd.Series:
    if target not in corr_df.columns:
        raise ValueError(f"Target {target} not in correlation matrix.")
        
    target_corr = corr_df[target].drop(target)
    return target_corr.abs().sort_values(ascending=False).head(n)

def coefficient_interpretation(coef_df: pd.DataFrame) -> list[str]:
    interpretations = []
    for _, row in coef_df.iterrows():
        feat = row['Feature']
        coef = row['Coefficient']
        if coef > 0:
            interpretations.append(f"A 1-unit increase in {feat} is associated with an estimated increase of {coef:.2f} in Sales.")
        elif coef < 0:
            interpretations.append(f"A 1-unit increase in {feat} is associated with an estimated decrease of {abs(coef):.2f} in Sales.")
        else:
            interpretations.append(f"{feat} does not appear to have a linear association with Sales.")
    return interpretations
