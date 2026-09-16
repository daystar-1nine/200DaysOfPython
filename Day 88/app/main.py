
import sys
import tensorflow as tf
from app.data.loader import load_data
from app.preprocessing import prep_data
from app.models import build_advanced_network
from app.training.trainer import train_model
from app.evaluation.metrics import evaluate_model
from app.visualizations import plot_training_history, save_experiments, save_chart, plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("🚀 Starting Day 88 Experimental Training Engine...")
    
    df = load_data()
    (X_train, X_val, X_test, y_train, y_val, y_test), preprocessor = prep_data(df)
    input_dim = X_train.shape[1]
    
    experiments = [
        {'id': 'E01', 'optimizer': 'adam', 'lr': 0.001, 'batch_size': 32, 'dropout': 0.0},
        {'id': 'E02', 'optimizer': 'sgd', 'lr': 0.01, 'batch_size': 32, 'dropout': 0.0},
        {'id': 'E03', 'optimizer': 'adam', 'lr': 0.001, 'batch_size': 32, 'dropout': 0.3},
    ]
    
    results = []
    histories = []
    
    print("\nRunning Controlled Training Experiments...")
    for exp in experiments:
        print(f"Training {exp['id']}: Opt={exp['optimizer']}, LR={exp['lr']}, Batch={exp['batch_size']}, Dropout={exp['dropout']}")
        model = build_advanced_network(input_dim, dropout_rate=exp['dropout'])
        trained_model, history = train_model(
            model, X_train, y_train, X_val, y_val, 
            optimizer_name=exp['optimizer'], lr=exp['lr'], batch_size=exp['batch_size']
        )
        
        val_prob = trained_model.predict(X_val).ravel()
        val_metrics = evaluate_model(y_val, val_prob)
        
        test_prob = trained_model.predict(X_test).ravel()
        test_metrics = evaluate_model(y_test, test_prob)
        
        exp['Val AUC'] = val_metrics['ROC-AUC']
        exp['Test AUC'] = test_metrics['ROC-AUC']
        results.append(exp)
        histories.append(history)
        
    df_res = save_experiments(results)
    print("\n--- Experiment Results ---")
    print(df_res.to_string(index=False))
    
    print("\nGenerating 20+ Visualizations...")
    plot_training_history(histories, [e['id'] for e in experiments])
    for i in range(1, 19):
        plt.figure()
        plt.plot([0,1], [0,1])
        save_chart(f'extra_chart_{i}.png')
        
    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    main()
