
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from app.config import Config

os.makedirs(Config.CHARTS_DIR, exist_ok=True)

def save_chart(name):
    plt.tight_layout()
    plt.savefig(os.path.join(Config.CHARTS_DIR, name), dpi=300)
    plt.close()

def plot_decision_boundary(clf, X, y, title, filename):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].min() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(title)
    save_chart(filename)

def generate_make_moons_boundaries():
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    
    # Linear
    clf_lin = SVC(kernel="linear").fit(X, y)
    plot_decision_boundary(clf_lin, X, y, 'Linear SVM Boundary', 'linear_boundary.png')
    
    # RBF
    clf_rbf = SVC(kernel="rbf").fit(X, y)
    plot_decision_boundary(clf_rbf, X, y, 'RBF SVM Boundary', 'rbf_boundary.png')
    
    # Poly
    clf_poly = SVC(kernel="poly", degree=3).fit(X, y)
    plot_decision_boundary(clf_poly, X, y, 'Polynomial SVM Boundary', 'polynomial_boundary.png')
