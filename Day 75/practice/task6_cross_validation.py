from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def cv_evaluate(X, y):
    return cross_val_score(LinearRegression(), X, y, cv=5).mean()
