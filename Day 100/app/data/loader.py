
import pandas as pd
import numpy as np
import os

def generate_mock_dataset(path):
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'age': np.random.normal(40, 15, n).astype(int),
        'income': np.random.normal(60000, 20000, n),
        'score': np.random.uniform(0, 100, n),
        'target': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    
    # Inject some noise
    df.loc[10:20, 'income'] = np.nan
    df.loc[100:105, 'age'] = -50 # outlier
    
    df.to_csv(path, index=False)
    return df

def load_and_clean(path):
    df = pd.read_csv(path)
    # Basic cleaning
    df['income'] = df['income'].fillna(df['income'].median())
    df = df[df['age'] > 0]
    return df
