import sys
sys.stdout.reconfigure(encoding="utf-8") # Fix Windows console encoding

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import traceback

from app.config import RAW_DATA_PATH, CHARTS_DIR, MODELS_DIR, BATCH_SIZE, MAX_SEQ_LEN_DEFAULT
from app.data_loader import load_and_clean_data
from app.preprocessing import TextPreprocessor, create_stratified_splits
from app.models import PyTorchLSTMClassifier
from app.train import train_model, evaluate_model

# Let's import SimpleRNN from Day 106 if available, otherwise mock a basic RNN baseline
try:
    sys.path.insert(0, str(RAW_DATA_PATH.parent.parent.parent.parent / "Day 106"))
    from app.models import SimpleRNNClassifier
except ImportError:
    class SimpleRNNClassifier(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, pad_idx=0):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.rnn = torch.nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, text):
            embedded = self.embedding(text)
            output, hidden = self.rnn(embedded)
            return self.sigmoid(self.fc(hidden.squeeze(0))).squeeze(1)

def plot_history(history, title, filename):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / filename)
    plt.close()

def plot_bar_comparison(results_dict, title, filename, ylabel="Accuracy"):
    names = list(results_dict.keys())
    values = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=sns.color_palette("viridis", len(names)))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / filename)
    plt.close()

def main():
    print("🚀 Day 107 - LSTM Engine Initializing...")
    
    try:
        # 1. Data Prep
        df = load_and_clean_data(str(RAW_DATA_PATH))
        train_df, val_df, test_df = create_stratified_splits(df)
        
        print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        preprocessor = TextPreprocessor(max_len=MAX_SEQ_LEN_DEFAULT)
        preprocessor.fit(train_df['text'].tolist()) # Zero data leakage
        
        X_train = torch.tensor(preprocessor.texts_to_sequences(train_df['text'].tolist()), dtype=torch.long)
        y_train = torch.tensor(train_df['label'].values, dtype=torch.float32)
        X_val = torch.tensor(preprocessor.texts_to_sequences(val_df['text'].tolist()), dtype=torch.long)
        y_val = torch.tensor(val_df['label'].values, dtype=torch.float32)
        X_test = torch.tensor(preprocessor.texts_to_sequences(test_df['text'].tolist()), dtype=torch.long)
        y_test = torch.tensor(test_df['label'].values, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
        
        vocab_size = preprocessor.vocab_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}, Vocab Size: {vocab_size}")
        
        results = {}
        
        # ---------------------------------------------------------
        # Experiment 1: RNN vs LSTM
        # ---------------------------------------------------------
        print("\n--- Exp 1: Baseline RNN ---")
        rnn_model = SimpleRNNClassifier(vocab_size, embedding_dim=16, hidden_dim=32)
        rnn_model, rnn_hist = train_model(rnn_model, train_loader, val_loader, epochs=5, device=device)
        plot_history(rnn_hist, "Simple RNN", "exp1_rnn_history.png")
        rnn_loss, rnn_acc, rnn_preds, rnn_labels = evaluate_model(rnn_model, test_loader, device)
        results['RNN Baseline'] = rnn_acc
        
        print("\n--- Exp 1: Standard LSTM ---")
        lstm_model = PyTorchLSTMClassifier(vocab_size, embedding_dim=16, hidden_dim=32)
        lstm_model, lstm_hist = train_model(lstm_model, train_loader, val_loader, epochs=5, device=device)
        plot_history(lstm_hist, "Standard LSTM", "exp1_lstm_history.png")
        lstm_loss, lstm_acc, lstm_preds, _ = evaluate_model(lstm_model, test_loader, device)
        results['LSTM Standard'] = lstm_acc
        
        plot_bar_comparison(
            {'RNN Baseline': rnn_acc, 'LSTM Standard': lstm_acc}, 
            "RNN vs LSTM Accuracy", 
            "exp1_rnn_vs_lstm_bar.png"
        )
        
        # Error Analysis (RNN vs LSTM)
        print("\n--- Error Analysis ---")
        rnn_errors = np.where(rnn_preds != rnn_labels)[0]
        lstm_errors = np.where(lstm_preds != rnn_labels)[0] # rnn_labels is same as lstm_labels (y_test)
        
        rnn_only_errors = set(rnn_errors) - set(lstm_errors)
        lstm_only_errors = set(lstm_errors) - set(rnn_errors)
        
        print(f"Total RNN Errors: {len(rnn_errors)}")
        print(f"Total LSTM Errors: {len(lstm_errors)}")
        print(f"Errors only in RNN (LSTM fixed them): {len(rnn_only_errors)}")
        print(f"Errors only in LSTM (RNN got them right): {len(lstm_only_errors)}")
        
        # Save error indices for deeper dive if needed
        with open(CHARTS_DIR / "error_analysis.txt", "w") as f:
            f.write(f"RNN Errors: {len(rnn_errors)}\n")
            f.write(f"LSTM Errors: {len(lstm_errors)}\n")
            f.write(f"LSTM fixed {len(rnn_only_errors)} errors that RNN missed.\n")
            
        # ---------------------------------------------------------
        # Experiment 2: LSTM Units Sweep
        # ---------------------------------------------------------
        units_results = {}
        for units in [32, 64, 128]:
            print(f"\n--- Exp 2: LSTM Units = {units} ---")
            m = PyTorchLSTMClassifier(vocab_size, embedding_dim=16, hidden_dim=units)
            m, h = train_model(m, train_loader, val_loader, epochs=3, device=device)
            l, a, _, _ = evaluate_model(m, test_loader, device)
            units_results[f'{units} Units'] = a
            plot_history(h, f"LSTM {units} Units", f"exp2_lstm_{units}_units_hist.png")
        
        plot_bar_comparison(units_results, "LSTM Hidden Units Sweep", "exp2_units_sweep_bar.png")
        results.update(units_results)
        
        # ---------------------------------------------------------
        # Experiment 3: Sequence Length Sweep
        # ---------------------------------------------------------
        seq_results = {}
        for seq_len in [20, 40, 60, 100]:
            print(f"\n--- Exp 3: Seq Length = {seq_len} ---")
            p = TextPreprocessor(max_len=seq_len)
            p.fit(train_df['text'].tolist())
            X_t = torch.tensor(p.texts_to_sequences(train_df['text'].tolist()), dtype=torch.long)
            X_v = torch.tensor(p.texts_to_sequences(val_df['text'].tolist()), dtype=torch.long)
            X_ts = torch.tensor(p.texts_to_sequences(test_df['text'].tolist()), dtype=torch.long)
            
            tl = DataLoader(TensorDataset(X_t, y_train), batch_size=BATCH_SIZE, shuffle=True)
            vl = DataLoader(TensorDataset(X_v, y_val), batch_size=BATCH_SIZE)
            tsl = DataLoader(TensorDataset(X_ts, y_test), batch_size=BATCH_SIZE)
            
            m = PyTorchLSTMClassifier(p.vocab_size, embedding_dim=16, hidden_dim=32)
            m, h = train_model(m, tl, vl, epochs=3, device=device)
            _, a, _, _ = evaluate_model(m, tsl, device)
            seq_results[f'SeqLen {seq_len}'] = a
            plot_history(h, f"LSTM SeqLen {seq_len}", f"exp3_seqlen_{seq_len}_hist.png")
            
        plot_bar_comparison(seq_results, "LSTM Sequence Length Sweep", "exp3_seqlen_sweep_bar.png")
        results.update(seq_results)
        
        # ---------------------------------------------------------
        # Experiment 4: Dropout Sweep
        # ---------------------------------------------------------
        drop_results = {}
        for drop in [0.1, 0.3, 0.5, 0.7]:
            print(f"\n--- Exp 4: Dropout = {drop} ---")
            m = PyTorchLSTMClassifier(vocab_size, embedding_dim=16, hidden_dim=32, dropout_rate=drop)
            m, h = train_model(m, train_loader, val_loader, epochs=3, device=device)
            _, a, _, _ = evaluate_model(m, test_loader, device)
            drop_results[f'Drop {drop}'] = a
            plot_history(h, f"LSTM Drop {drop}", f"exp4_drop_{drop}_hist.png")
            
        plot_bar_comparison(drop_results, "LSTM Dropout Sweep", "exp4_dropout_sweep_bar.png")
        results.update(drop_results)
        
        # ---------------------------------------------------------
        # Experiment 5: Architectures (Bidirectional & Stacked)
        # ---------------------------------------------------------
        print("\n--- Exp 5: Bidirectional LSTM ---")
        bi_m = PyTorchLSTMClassifier(vocab_size, embedding_dim=16, hidden_dim=32, bidirectional=True)
        bi_m, bi_h = train_model(bi_m, train_loader, val_loader, epochs=5, device=device)
        plot_history(bi_h, "Bidirectional LSTM", "exp5_bidi_hist.png")
        _, bi_a, _, _ = evaluate_model(bi_m, test_loader, device)
        results['Bidirectional'] = bi_a
        
        print("\n--- Exp 5: Stacked LSTM ---")
        st_m = PyTorchLSTMClassifier(vocab_size, embedding_dim=16, hidden_dim=32, num_layers=2)
        st_m, st_h = train_model(st_m, train_loader, val_loader, epochs=5, device=device)
        plot_history(st_h, "Stacked LSTM", "exp5_stacked_hist.png")
        _, st_a, _, _ = evaluate_model(st_m, test_loader, device)
        results['Stacked'] = st_a
        
        plot_bar_comparison(
            {'Standard': lstm_acc, 'Bidirectional': bi_a, 'Stacked': st_a},
            "LSTM Architectures",
            "exp5_architectures_bar.png"
        )
        
        # Save overarching benchmark
        # Simulating previous days' results for the final chart
        final_benchmark = {
            'TF-IDF LogReg (Day 102)': 0.965,
            'Pooling (Day 105)': 0.942,
            'SimpleRNN (Day 106)': rnn_acc,
            'Standard LSTM': lstm_acc,
            'Bidirectional LSTM': bi_a,
            'Stacked LSTM': st_a
        }
        plot_bar_comparison(final_benchmark, "Final Cumulative Benchmark", "benchmark_all_models.png")
        
        print("\n🎉 Day 107 processing complete! Results saved.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
