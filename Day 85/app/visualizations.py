
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from app.config import Config

os.makedirs(Config.CHARTS_DIR, exist_ok=True)

def save_chart(name):
    plt.tight_layout()
    plt.savefig(os.path.join(Config.CHARTS_DIR, name), dpi=300)
    plt.close()

def plot_cm(cm, labels, title, filename):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    save_chart(filename)
