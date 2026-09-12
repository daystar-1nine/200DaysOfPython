import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_train_test_split(df: pd.DataFrame, target_col: str, test_size: float = 0.20, random_state: int = 42):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
