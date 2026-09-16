
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from app.config import Config

os.makedirs(Config.CHARTS_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def save_chart(name):
    plt.tight_layout()
    plt.savefig(os.path.join(Config.CHARTS_DIR, name), dpi=300)
    plt.close()

def plot_training_history(histories, names):
    # Just plot the first one for demonstration
    if histories:
        history = histories[0]
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        save_chart('training_loss.png')
        
        plt.figure()
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Val AUC')
        plt.title('Training and Validation AUC')
        plt.legend()
        save_chart('training_auc.png')

def save_experiments(results):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(Config.OUTPUT_DIR, 'experiment_results.csv'), index=False)
    return df
