
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def get_preprocessor(df):
    X = df.drop('Churn', axis=1)
    
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    return preprocessor, X, df['Churn']

def prepare_data(df, test_size=0.2, random_state=42):
    preprocessor, X, y = get_preprocessor(df)
    y = y.map({'Yes': 1, 'No': 0}) if y.dtype == 'object' else y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Store feature names
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_cols = list(cat_encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns))
    feature_names = num_cols + cat_cols
    
    return X_train_prep, X_test_prep, y_train, y_test, feature_names, preprocessor
