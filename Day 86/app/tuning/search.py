
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def tune_svm(preprocessor, X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(probability=True, random_state=42))
    ])
    
    # We use a small grid for speed since SVM can be expensive
    param_grid = {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1, 10],
        "classifier__gamma": ["scale", 0.1]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
