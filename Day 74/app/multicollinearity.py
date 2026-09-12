import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import numpy as np

def calculate_vif(df: pd.DataFrame, numeric_features: list[str] = None) -> pd.DataFrame:
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_features].dropna()
    if X.empty:
        return pd.DataFrame(columns=['Feature', 'VIF'])

    X = add_constant(X, has_constant='add')

    vif_data = []
    for i in range(X.shape[1]):
        col_name = X.columns[i]
        if col_name == 'const':
            continue
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({
            'Feature': col_name,
            'VIF': float(vif)
        })

    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values(by='VIF', ascending=False).reset_index(drop=True)
    return vif_df

def flag_high_vif(vif_or_df: pd.DataFrame, threshold: float = 10.0) -> list[str]:
    if 'VIF' not in vif_or_df.columns:
        vif_df = calculate_vif(vif_or_df)
    else:
        vif_df = vif_or_df
    high_vif = vif_df[vif_df['VIF'] > threshold]
    return high_vif['Feature'].tolist()
