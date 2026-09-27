import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, OUTPUT_DIR, CHARTS_DIR,
    SEED, MAX_VOCAB_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM, HIDDEN_DIM,
    DENSE_DIM, DROPOUT_RATE, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    EARLY_STOPPING_PATIENCE, DEVICE
)
from app.data import load_raw_data, clean_sms_data, validate_sms_data, split_dataset
from app.preprocessing import tokenize_text, Vocabulary, encode_texts, pad_sequences
from app.models import (
    SimpleRNNClassifier, LSTMClassifier, GRUClassifier,
    BiGRUClassifier, BiLSTMClassifier, StackedGRUClassifier
)
from app.training.trainer import fit_model, evaluate_epoch
from app.evaluation import (
    compute_classification_metrics, analyze_thresholds,
    compute_roc_curve_data, compute_pr_curve_data
)
from app.analysis import (
    perform_error_analysis, create_model_comparison_table,
    compute_model_efficiency
)
from app.visualization import (
    generate_dataset_plots, generate_training_plots,
    generate_evaluation_plots, generate_benchmark_plots
)

def create_dataloader(padded_seqs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(padded_seqs, dtype=torch.long)
    tensor_y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def run_controlled_benchmark():
    """
    Main benchmark runner executing Experiments 1-6 in a fully reproducible environment.
    """
    print(f"Setting random seed to {SEED}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    print(f"Using device: {DEVICE}")
    
    # 1. Load, clean, validate
    raw_df = load_raw_data(RAW_DATA_PATH)
    clean_df = clean_sms_data(raw_df)
    validate_sms_data(clean_df)
    
    # Dataset plots (Plots 1-4)
    print("Generating dataset exploratory plots...")
    generate_dataset_plots(clean_df, CHARTS_DIR)
    
    # 2. Stratified Split
    train_df, val_df, test_df = split_dataset(clean_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=SEED)
    print(f"Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 3. Fit Vocabulary on Train Set ONLY (Zero Leakage)
    vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
    train_tokens = [tokenize_text(t) for t in train_df["text"]]
    vocab.fit(train_tokens)
    vocab_size = len(vocab)
    print(f"Fitted Vocabulary Size (Train only): {vocab_size}")
    
    # 4. Encode & Pad
    train_encoded = encode_texts(train_df["text"], vocab, tokenize_text)
    val_encoded = encode_texts(val_df["text"], vocab, tokenize_text)
    test_encoded = encode_texts(test_df["text"], vocab, tokenize_text)
    
    X_train = pad_sequences(train_encoded, max_len=MAX_SEQ_LENGTH)
    X_val = pad_sequences(val_encoded, max_len=MAX_SEQ_LENGTH)
    X_test = pad_sequences(test_encoded, max_len=MAX_SEQ_LENGTH)
    
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    
    train_loader = create_dataloader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = torch.nn.BCELoss()
    
    # Data storage
    primary_models = {}
    histories = {}
    test_probs = {}
    test_metrics = {}
    comparison_records = []
    
    # -------------------------------------------------------------
    # Experiment 1: Core Comparison (RNN vs LSTM vs GRU)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 1: RNN vs LSTM vs GRU Controlled Comparison")
    print("="*50)
    
    model_configs = [
        ("RNN", SimpleRNNClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE)),
        ("LSTM", LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE)),
        ("GRU", GRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE))
    ]
    
    for name, model in model_configs:
        print(f"\n--- Training {name} ---")
        history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=EARLY_STOPPING_PATIENCE,
            device=DEVICE,
            verbose=False
        )
        _, _, probs, _ = evaluate_epoch(model, test_loader, criterion, DEVICE)
        metrics = compute_classification_metrics(y_test, probs)
        
        primary_models[name] = model
        histories[name] = history
        test_probs[name] = probs
        test_metrics[name] = metrics
        
        params = model.count_parameters()
        record = {
            "model": name,
            "experiment": "Exp1_Core_Recurrent",
            "parameters": params,
            "training_time_sec": round(history["training_time_sec"], 3),
            "epochs_trained": history["epochs_trained"],
            **metrics
        }
        comparison_records.append(record)
        print(f"{name} Result -> F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Params: {params} | Time: {record['training_time_sec']}s")

    # Generate training plots (Plots 5-10)
    generate_training_plots(histories, CHARTS_DIR)

    # -------------------------------------------------------------
    # Experiment 2: Hidden Units Sweep (32, 64, 128)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 2: Hidden Units Sweep (32, 64, 128)")
    print("="*50)
    units_records = []
    
    for u in [32, 64, 128]:
        for arch_name, arch_cls in [("RNN", SimpleRNNClassifier), ("LSTM", LSTMClassifier), ("GRU", GRUClassifier)]:
            m = arch_cls(vocab_size, EMBEDDING_DIM, hidden_dim=u, dense_dim=DENSE_DIM, dropout_rate=DROPOUT_RATE)
            h = fit_model(m, train_loader, val_loader, epochs=8, lr=LEARNING_RATE, patience=2, device=DEVICE, verbose=False)
            _, _, p, _ = evaluate_epoch(m, test_loader, criterion, DEVICE)
            met = compute_classification_metrics(y_test, p)
            units_records.append({
                "architecture": arch_name,
                "units": u,
                "parameters": m.count_parameters(),
                "training_time_sec": round(h["training_time_sec"], 3),
                "f1": met["f1"],
                "roc_auc": met["roc_auc"],
                "accuracy": met["accuracy"]
            })
            print(f"Units Sweep: {arch_name}({u}) -> F1: {met['f1']:.4f}, Params: {m.count_parameters()}")
            
    units_df = pd.DataFrame(units_records)

    # -------------------------------------------------------------
    # Experiment 3: Sequence Length Sweep (20, 40, 60, 100)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 3: Sequence Length Sweep (20, 40, 60, 100)")
    print("="*50)
    seqlen_records = []
    
    for sl in [20, 40, 60, 100]:
        X_tr_sl = pad_sequences(train_encoded, max_len=sl)
        X_v_sl = pad_sequences(val_encoded, max_len=sl)
        X_te_sl = pad_sequences(test_encoded, max_len=sl)
        
        tr_l = create_dataloader(X_tr_sl, y_train, BATCH_SIZE, shuffle=True)
        v_l = create_dataloader(X_v_sl, y_val, BATCH_SIZE, shuffle=False)
        te_l = create_dataloader(X_te_sl, y_test, BATCH_SIZE, shuffle=False)
        
        m_sl = GRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE)
        h_sl = fit_model(m_sl, tr_l, v_l, epochs=8, lr=LEARNING_RATE, patience=2, device=DEVICE, verbose=False)
        _, _, p_sl, _ = evaluate_epoch(m_sl, te_l, criterion, DEVICE)
        met_sl = compute_classification_metrics(y_test, p_sl)
        
        seqlen_records.append({
            "seq_len": sl,
            "f1": met_sl["f1"],
            "roc_auc": met_sl["roc_auc"],
            "training_time_sec": round(h_sl["training_time_sec"], 3),
            "parameters": m_sl.count_parameters()
        })
        print(f"SeqLen {sl} -> F1: {met_sl['f1']:.4f}, Time: {h_sl['training_time_sec']:.2f}s")
        
    seqlen_df = pd.DataFrame(seqlen_records)

    # -------------------------------------------------------------
    # Experiment 4: Dropout Sweep (0.0, 0.2, 0.3, 0.5)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 4: GRU Dropout Sweep (0.0, 0.2, 0.3, 0.5)")
    print("="*50)
    dropout_records = []
    
    for dr in [0.0, 0.2, 0.3, 0.5]:
        m_dr = GRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, dropout_rate=dr)
        h_dr = fit_model(m_dr, train_loader, val_loader, epochs=8, lr=LEARNING_RATE, patience=2, device=DEVICE, verbose=False)
        _, _, p_dr, _ = evaluate_epoch(m_dr, test_loader, criterion, DEVICE)
        met_dr = compute_classification_metrics(y_test, p_dr)
        
        dropout_records.append({
            "dropout": dr,
            "f1": met_dr["f1"],
            "roc_auc": met_dr["roc_auc"],
            "accuracy": met_dr["accuracy"]
        })
        print(f"Dropout {dr} -> F1: {met_dr['f1']:.4f}")
        
    dropout_df = pd.DataFrame(dropout_records)

    # -------------------------------------------------------------
    # Experiment 5: Bidirectional GRU vs BiLSTM
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 5: Bidirectional GRU & BiLSTM")
    print("="*50)
    
    bigru = BiGRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE)
    h_bigru = fit_model(bigru, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE, device=DEVICE, verbose=False)
    _, _, p_bigru, _ = evaluate_epoch(bigru, test_loader, criterion, DEVICE)
    met_bigru = compute_classification_metrics(y_test, p_bigru)
    test_probs["BiGRU"] = p_bigru
    
    comparison_records.append({
        "model": "BiGRU",
        "experiment": "Exp5_Bidirectional",
        "parameters": bigru.count_parameters(),
        "training_time_sec": round(h_bigru["training_time_sec"], 3),
        "epochs_trained": h_bigru["epochs_trained"],
        **met_bigru
    })
    print(f"BiGRU Result -> F1: {met_bigru['f1']:.4f} | ROC-AUC: {met_bigru['roc_auc']:.4f} | Params: {bigru.count_parameters()}")

    # -------------------------------------------------------------
    # Experiment 6: Stacked GRU
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 6: Stacked GRU (64 -> 32)")
    print("="*50)
    
    stacked_gru = StackedGRUClassifier(vocab_size, EMBEDDING_DIM, hidden_dim1=64, hidden_dim2=32, dropout_rate=DROPOUT_RATE)
    h_stacked = fit_model(stacked_gru, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE, device=DEVICE, verbose=False)
    _, _, p_stacked, _ = evaluate_epoch(stacked_gru, test_loader, criterion, DEVICE)
    met_stacked = compute_classification_metrics(y_test, p_stacked)
    test_probs["StackedGRU"] = p_stacked
    
    comparison_records.append({
        "model": "StackedGRU",
        "experiment": "Exp6_Stacked",
        "parameters": stacked_gru.count_parameters(),
        "training_time_sec": round(h_stacked["training_time_sec"], 3),
        "epochs_trained": h_stacked["epochs_trained"],
        **met_stacked
    })
    print(f"StackedGRU Result -> F1: {met_stacked['f1']:.4f} | ROC-AUC: {met_stacked['roc_auc']:.4f} | Params: {stacked_gru.count_parameters()}")

    # -------------------------------------------------------------
    # Output File Generation & ROC/PR computation
    # -------------------------------------------------------------
    comparison_df = create_model_comparison_table(comparison_records)
    efficiency_df = compute_model_efficiency(comparison_df)
    
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    efficiency_df.to_csv(OUTPUT_DIR / "efficiency.csv", index=False)
    
    # All experiment sweeps to experiments.csv
    exp_summary = pd.concat([
        units_df.assign(experiment="Exp2_HiddenUnits"),
        seqlen_df.assign(experiment="Exp3_SeqLength"),
        dropout_df.assign(experiment="Exp4_Dropout")
    ], ignore_index=True)
    exp_summary.to_csv(OUTPUT_DIR / "experiments.csv", index=False)
    
    # Threshold Analysis on primary GRU
    print("\nRunning Threshold Analysis on GRU...")
    thresh_df = analyze_thresholds(
        y_test, test_probs["GRU"],
        model_name="GRU",
        save_path=OUTPUT_DIR / "threshold_analysis.csv"
    )
    
    # Error Analysis
    print("Performing message-level error analysis...")
    error_analysis_res = perform_error_analysis(
        test_texts=test_df["text"].tolist(),
        y_true=y_test,
        model_predictions={m: (p >= 0.5).astype(int) for m, p in test_probs.items() if m in ["RNN", "LSTM", "GRU"]},
        model_probabilities={m: p for m, p in test_probs.items() if m in ["RNN", "LSTM", "GRU"]},
        output_dir=OUTPUT_DIR
    )
    
    # ROC and PR Curve data
    roc_data = {m: compute_roc_curve_data(y_test, p) for m, p in test_probs.items()}
    pr_data = {m: compute_pr_curve_data(y_test, p) for m, p in test_probs.items()}
    
    # Generate Evaluation plots (Plots 11-15)
    generate_evaluation_plots(y_test, test_probs, roc_data, pr_data, CHARTS_DIR)
    
    # Generate Benchmark plots (Plots 16-22)
    generate_benchmark_plots(comparison_df, units_df, seqlen_df, dropout_df, CHARTS_DIR)
    
    print("\n" + "="*50)
    print("Controlled Benchmark Complete!")
    print(f"Generated 22 Visualizations in {CHARTS_DIR}")
    print("="*50)
    
    return {
        "comparison_df": comparison_df,
        "efficiency_df": efficiency_df,
        "error_analysis": error_analysis_res
    }
