
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.config import Config

os.makedirs(Config.CHARTS_DIR, exist_ok=True)

def save_chart(name):
    plt.tight_layout()
    plt.savefig(os.path.join(Config.CHARTS_DIR, name), dpi=300)
    plt.close()

def plot_history(history):
    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    save_chart('training_loss_curve.png')
    
    # Plot AUC
    plt.figure()
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    save_chart('training_auc_curve.png')
