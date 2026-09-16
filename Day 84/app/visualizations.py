
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from app.config import Config

os.makedirs(Config.CHARTS_DIR, exist_ok=True)

def save_chart(name):
    plt.tight_layout()
    plt.savefig(os.path.join(Config.CHARTS_DIR, name), dpi=300)
    plt.close()

def plot_pca(X_train_scaled, y_train):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter, label='Churn')
    plt.title('PCA Projection of Churn Data (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    save_chart('pca_projection.png')

def generate_evaluation_charts(results_df):
    metrics = ['ROC-AUC', 'F1', 'Average Precision']
    for m in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='Model', y=m, hue='Model', palette='viridis', legend=False)
        plt.title(f'{m} Comparison')
        plt.xticks(rotation=45)
        save_chart(f'{m.lower().replace("-", "_").replace(" ", "_")}_comparison.png')
        
def plot_k_vs_score(X_train_scaled, y_train, X_test_scaled, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    k_vals = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    scores = []
    
    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_prob = knn.predict_proba(X_test_scaled)[:, 1]
        scores.append(roc_auc_score(y_test, y_prob))
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, scores, marker='o', linestyle='-', color='b')
    plt.title('K vs ROC-AUC Score (Test Data)')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('ROC-AUC Score')
    save_chart('k_vs_cv_score.png')
