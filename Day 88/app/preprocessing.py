
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
    
    # 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    
    return (X_train_t, X_val_t, X_test_t, y_train, y_val, y_test), preprocessor
