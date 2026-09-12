from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def compare_models(X, y):
    lin_reg = LinearRegression().fit(X, y)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_reg = LinearRegression().fit(X_poly, y)
    return lin_reg.score(X, y), poly_reg.score(X_poly, y)
