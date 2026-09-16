
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.inspection import permutation_importance
from app.config import Config

os.makedirs(Config.CHARTS_DIR, exist_ok=True)

def save_chart(name):
    plt.tight_layout()
    plt.savefig(os.path.join(Config.CHARTS_DIR, name), dpi=300)
    plt.close()

def plot_model_comparison(results_df):
    metrics = ['ROC AUC', 'Average Precision', 'F1 Score', 'Recall']
    for m in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='Model', y=m, hue='Model', palette='viridis', legend=False)
        plt.title(f'Model Comparison: {m}')
        plt.xticks(rotation=45)
        save_chart(f'comparison_{m.lower().replace(" ", "_")}.png')

def plot_feature_importance(model, feature_names, name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:10]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[idx], y=[feature_names[i] for i in idx], hue=[feature_names[i] for i in idx], legend=False, palette='mako')
        plt.title(f'{name} Feature Importance')
        save_chart(f'feature_importance_{name.lower().replace(" ", "_")}.png')

def plot_permutation_importance(model, X_test, y_test, feature_names, name):
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    idx = result.importances_mean.argsort()[::-1][:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=result.importances_mean[idx], y=[feature_names[i] for i in idx], hue=[feature_names[i] for i in idx], legend=False, palette='rocket')
    plt.title(f'{name} Permutation Importance')
    save_chart(f'permutation_importance_{name.lower().replace(" ", "_")}.png')

def plot_experiments(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier
    # Learning rate
    lrs = [0.01, 0.05, 0.1, 0.2]
    scores = []
    for lr in lrs:
        m = XGBClassifier(learning_rate=lr, n_estimators=100, random_state=42, eval_metric='logloss')
        m.fit(X_train, y_train)
        scores.append(m.score(X_test, y_test))
    plt.figure()
    plt.plot(lrs, scores, marker='o')
    plt.title("XGBoost: Learning Rate vs Accuracy")
    save_chart('xgb_learning_rate.png')
    
    # Depth
    depths = [2, 3, 4, 6]
    scores = []
    for d in depths:
        m = XGBClassifier(max_depth=d, n_estimators=100, random_state=42, eval_metric='logloss')
        m.fit(X_train, y_train)
        scores.append(m.score(X_test, y_test))
    plt.figure()
    plt.plot(depths, scores, marker='o')
    plt.title("XGBoost: Max Depth vs Accuracy")
    save_chart('xgb_max_depth.png')
