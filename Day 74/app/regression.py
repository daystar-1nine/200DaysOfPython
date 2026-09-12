import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

try:
    from app.config import AppConfig
except ImportError:
    from config import AppConfig

def train_model(X, y) -> LinearRegression:
    """
    Fits and returns a standard LinearRegression model.
    """
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

def get_coefficients(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extracts sorted coefficients DataFrame from model.
    """
    if hasattr(model, 'get_coefficients'):
        return model.get_coefficients(feature_names)
    coefs = getattr(model, 'coef_', [])
    df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    df['Abs_Coefficient'] = df['Coefficient'].abs()
    df = df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
    return df.reset_index(drop=True)

def evaluate_model(model, X, y) -> dict:
    """
    Computes performance metrics on evaluation dataset.
    """
    preds = model.predict(X)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        'MAE': float(mean_absolute_error(y, preds)),
        'MSE': float(mean_squared_error(y, preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y, preds))),
        'R2': float(r2_score(y, preds))
    }

class MultipleLinearRegressor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = self._build_full_pipeline()
        
    def _build_full_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config.NUMERIC_FEATURES),
                ('cat', categorical_transformer, self.config.CATEGORICAL_FEATURES)
            ],
            remainder='passthrough'
        )
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.pipeline.fit(X_train, y_train)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)
        
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.pipeline.score(X, y)
        
    def get_coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        coefs = self.pipeline.named_steps['regressor'].coef_
        df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
        df['Abs_Coefficient'] = df['Coefficient'].abs()
        df = df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
        return df.reset_index(drop=True)
        
    def get_intercept(self) -> float:
        return self.pipeline.named_steps['regressor'].intercept_
        
    def equation_summary(self) -> str:
        intercept = self.get_intercept()
        
        preprocessor = self.pipeline.named_steps['preprocessor']
        num_names = self.config.NUMERIC_FEATURES.copy()
        try:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_names = cat_encoder.get_feature_names_out(self.config.CATEGORICAL_FEATURES).tolist()
        except:
            cat_names = []
        
        feature_names = num_names + cat_names
        
        coef_df = self.get_coefficients(feature_names)
        
        eq = f"Sales = {intercept:.2f}"
        for _, row in coef_df.iterrows():
            feat = row['Feature']
            c = row['Coefficient']
            sign = "+" if c >= 0 else "-"
            eq += f" {sign} {abs(c):.2f}*{feat}"
            
        return eq
