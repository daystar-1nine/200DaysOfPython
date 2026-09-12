from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from app.preprocessing import build_preprocessor
from app.config import AppConfig

def tune_ridge(X, y, alphas=None, cv=5):
    config = AppConfig()
    alphas = alphas or config.ALPHAS
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)),
        ('regressor', Ridge(random_state=42))
    ])
    
    param_grid = {'regressor__alpha': alphas}
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_root_mean_squared_error')
    grid.fit(X, y)
    
    return grid, {"best_alpha": grid.best_params_['regressor__alpha'], "best_rmse": -grid.best_score_}

def tune_lasso(X, y, alphas=None, cv=5):
    config = AppConfig()
    alphas = alphas or config.ALPHAS
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)),
        ('regressor', Lasso(max_iter=5000, random_state=42))
    ])
    
    param_grid = {'regressor__alpha': alphas}
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_root_mean_squared_error')
    grid.fit(X, y)
    
    return grid, {"best_alpha": grid.best_params_['regressor__alpha'], "best_rmse": -grid.best_score_}

def tune_polynomial_degree(X, y, max_degree=5, cv=5):
    from sklearn.linear_model import LinearRegression
    config = AppConfig()
    
    best_degree = 1
    best_score = float('inf')
    results = []
    
    for degree in range(1, max_degree + 1):
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', LinearRegression())
        ])
        from sklearn.model_selection import cross_val_score
        scores = -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        mean_score = scores.mean()
        
        results.append({"degree": degree, "rmse": mean_score})
        if mean_score < best_score:
            best_score = mean_score
            best_degree = degree
            
    return best_degree, best_score, results

def run_hyperparameter_tuning(X, y, config=None) -> dict:
    if config is None:
        config = AppConfig()
        
    _, ridge_res = tune_ridge(X, y, config.ALPHAS, config.CV_FOLDS)
    _, lasso_res = tune_lasso(X, y, config.ALPHAS, config.CV_FOLDS)
    best_deg, best_poly_score, poly_res = tune_polynomial_degree(X, y, config.MAX_POLY_DEGREE, config.CV_FOLDS)
    
    return {
        "ridge": ridge_res,
        "lasso": lasso_res,
        "polynomial": {"best_degree": best_deg, "best_rmse": best_poly_score, "results": poly_res}
    }
