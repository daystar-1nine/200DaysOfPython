import numpy as np
from sklearn.model_selection import cross_val_score

def run_cross_validation(model, X, y, cv=5) -> dict:
    estimator = getattr(model, 'pipeline', model)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    scores = -scores
    
    return {
        "cv_scores": scores.tolist(),
        "mean_cv_rmse": float(np.mean(scores)),
        "std_cv_rmse": float(np.std(scores))
    }
