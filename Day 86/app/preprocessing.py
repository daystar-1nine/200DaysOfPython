
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def prep_data(df):
    df_clean = df.copy()
    if 'CustomerID' in df_clean.columns: df_clean = df_clean.drop('CustomerID', axis=1)
    
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn'].map({'Yes': 1, 'No': 0}) if df_clean['Churn'].dtype == 'object' else df_clean['Churn']
    
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])
    
    # Stratified split to handle imbalance
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), preprocessor
