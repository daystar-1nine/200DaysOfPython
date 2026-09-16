
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def tune_xgboost(X_train, y_train):
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4],
        "subsample": [0.8, 1.0]
    }
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
