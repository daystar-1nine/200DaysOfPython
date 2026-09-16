
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator

# Helper to enforce dense output for GaussianNB
class ToDense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if hasattr(X, 'toarray') else X

def prep_churn_data(df):
    df_clean = df.copy()
    if 'CustomerID' in df_clean.columns: df_clean = df_clean.drop('CustomerID', axis=1)
    
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn'].map({'Yes': 1, 'No': 0}) if df_clean['Churn'].dtype == 'object' else df_clean['Churn']
    
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor

def prep_text_data(df):
    X = df['message']
    y = df['category']
    return train_test_split(X, y, test_size=0.2, random_state=42)
