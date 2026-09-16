
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from app.loader import AppConfig

CHARTS_DIR = os.path.join(AppConfig.BASE_DIR, 'output', 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, filename), dpi=300)
    plt.close()

def plot_model_comparison(results_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='Model', y=metric, hue='Model', palette='viridis', legend=False)
        plt.title(f'Model Comparison: {metric}')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        save_chart(f'model_comparison_{metric.lower().replace(" ", "_")}.png')

def plot_roc_curves(models_dict, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    save_chart('roc_curves_comparison.png')

def plot_pr_curves(models_dict, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    save_chart('pr_curves_comparison.png')

def plot_confusion_matrices(models_dict, X_test, y_test):
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        save_chart(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')

def plot_feature_importance(model, feature_names, top_n=10, model_name="Model"):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], hue=[feature_names[i] for i in indices], palette='viridis', legend=False)
        plt.title(f'Top {top_n} Feature Importances ({model_name})')
        save_chart(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        
def plot_learning_rate_sensitivity(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    lrs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    train_scores = []
    test_scores = []
    
    for lr in lrs:
        gb = GradientBoostingClassifier(learning_rate=lr, n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        train_scores.append(gb.score(X_train, y_train))
        test_scores.append(gb.score(X_test, y_test))
        
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, train_scores, label='Train Accuracy', marker='o')
    plt.plot(lrs, test_scores, label='Test Accuracy', marker='s')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Gradient Boosting Sensitivity to Learning Rate')
    plt.legend()
    save_chart('sensitivity_learning_rate.png')

def plot_n_estimators_sensitivity(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    estimators = [10, 50, 100, 200, 300]
    train_scores = []
    test_scores = []
    
    for n in estimators:
        gb = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        train_scores.append(gb.score(X_train, y_train))
        test_scores.append(gb.score(X_test, y_test))
        
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, train_scores, label='Train Accuracy', marker='o')
    plt.plot(estimators, test_scores, label='Test Accuracy', marker='s')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Gradient Boosting Sensitivity to n_estimators')
    plt.legend()
    save_chart('sensitivity_n_estimators.png')

def plot_tree_depth_sensitivity(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    depths = [1, 2, 3, 4, 5, 7, 10]
    train_scores = []
    test_scores = []
    
    for d in depths:
        gb = GradientBoostingClassifier(max_depth=d, n_estimators=50, random_state=42)
        gb.fit(X_train, y_train)
        train_scores.append(gb.score(X_train, y_train))
        test_scores.append(gb.score(X_test, y_test))
        
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, label='Train Accuracy', marker='o')
    plt.plot(depths, test_scores, label='Test Accuracy', marker='s')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Gradient Boosting Sensitivity to Max Depth')
    plt.legend()
    save_chart('sensitivity_max_depth.png')

def plot_business_cost_curve(thresholds, costs, optimal_t, optimal_c):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, lw=2)
    plt.axvline(optimal_t, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_t:.2f}')
    plt.plot(optimal_t, optimal_c, 'ro')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Estimated Business Cost ($)')
    plt.title('Business Cost vs Classification Threshold')
    plt.legend()
    save_chart('business_cost_curve.png')
