import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain features for tree and linear models."""
    df_feat = df.copy()
    
    # 1. Tenure to Age ratio
    df_feat['Tenure_to_Age_Ratio'] = df_feat['Tenure_Months'] / (df_feat['Age'] + 1e-5)
    
    # 2. Charges deviation (Monthly charge compared to average calculated monthly charge)
    expected_monthly = df_feat['Total_Charges'] / (df_feat['Tenure_Months'].replace(0, 1) + 1e-5)
    df_feat['Charges_Deviation'] = df_feat['Monthly_Charges'] - expected_monthly
    
    # 3. Support calls per month
    df_feat['Support_Per_Month'] = df_feat['Support_Calls'] / (df_feat['Tenure_Months'].replace(0, 1) + 1e-5)
    
    # 4. Composite high-risk indicator
    df_feat['High_Risk_Flag'] = (
        (df_feat['Support_Calls'] >= 4) | 
        (df_feat['Late_Payments'] >= 2) | 
        (df_feat['Complaints'] >= 2)
    ).astype(int)
    
    return df_feat
