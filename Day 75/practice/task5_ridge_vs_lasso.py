from sklearn.linear_model import Ridge, Lasso

def compare_regularization(X, y):
    r = Ridge(alpha=1.0).fit(X, y)
    l = Lasso(alpha=1.0).fit(X, y)
    return r.score(X, y), l.score(X, y)
