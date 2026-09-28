import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, OUTPUT_DIR, CHARTS_DIR, ATTENTION_EXAMPLES_DIR,
    SEED, MAX_VOCAB_SIZE, MAX_SEQ_LENGTH, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM,
    DENSE_DIM, DROPOUT_RATE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE, DEVICE
)
from app.data import load_raw_data, clean_sms_data, validate_sms_data, split_dataset
from app.preprocessing import tokenize_text, Vocabulary, encode_texts, pad_sequences
from app.models import (
    GRUClassifier, GRUAttentionClassifier,
    SimpleRNNClassifier, LSTMClassifier, TfidfLogisticRegressionBaseline
)
from app.training.trainer import fit_model, evaluate_epoch
from app.evaluation import (
    compute_classification_metrics, analyze_thresholds,
    compute_roc_curve_data, compute_pr_curve_data
)
from app.analysis import (
    compute_attention_entropy, summarize_attention_distribution,
    extract_top_attention_tokens, aggregate_token_attention,
    perform_error_analysis
)
from app.visualization import (
    generate_attention_example_heatmaps,
    generate_training_plots,
    generate_benchmark_visualizations
)

def create_dataloader(padded_seqs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(padded_seqs, dtype=torch.long)
    tensor_y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def run_attention_benchmark():
    print(f"Setting random seed to {SEED} (Reproducible Environment)")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    print(f"Hardware execution device: {DEVICE}")
    
    # 1. Load, clean, validate
    raw_df = load_raw_data(RAW_DATA_PATH)
    clean_df = clean_sms_data(raw_df)
    validate_sms_data(clean_df)
    
    # 2. Stratified split (70 / 15 / 15)
    train_df, val_df, test_df = split_dataset(clean_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=SEED)
    print(f"Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 3. Fit vocabulary exclusively on Train set (Zero Data Leakage)
    vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
    train_tokens = [tokenize_text(t) for t in train_df["text"]]
    vocab.fit(train_tokens)
    vocab_size = len(vocab)
    print(f"Fitted Vocabulary Size (Train only): {vocab_size}")
    
    # 4. Preprocessing
    train_encoded = encode_texts(train_df["text"], vocab, tokenize_text)
    val_encoded = encode_texts(val_df["text"], vocab, tokenize_text)
    test_encoded = encode_texts(test_df["text"], vocab, tokenize_text)
    test_tokenized = [tokenize_text(t) for t in test_df["text"]]
    
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
    
    # Storage
    histories = {}
    test_probs = {}
    comparison_records = []
    
    # -------------------------------------------------------------
    # Main Benchmark: Section 26 & 27
    # Models: TF-IDF+LR, SimpleRNN, LSTM, GRU, GRU+Attention
    # -------------------------------------------------------------
    print("\n" + "="*55)
    print("MAIN BENCHMARK: Recurrent Architectures vs GRU + Attention")
    print("="*55)
    
    # Model A: TF-IDF + Logistic Regression
    print("\n--- Model A: TF-IDF + Logistic Regression ---")
    tfidf_model = TfidfLogisticRegressionBaseline()
    tfidf_model.fit(train_df["text"], y_train)
    tfidf_probs = tfidf_model.predict_proba(test_df["text"])
    tfidf_metrics = compute_classification_metrics(y_test, tfidf_probs)
    test_probs["TF-IDF + LR"] = tfidf_probs
    comparison_records.append({
        "model": "TF-IDF + LR",
        "parameters": tfidf_model.count_parameters(),
        "training_time_sec": round(tfidf_model.training_time_sec, 3),
        "epochs_trained": 1,
        **tfidf_metrics
    })
    print(f"TF-IDF + LR -> F1: {tfidf_metrics['f1']:.4f} | ROC-AUC: {tfidf_metrics['roc_auc']:.4f} | Time: {tfidf_model.training_time_sec:.3f}s")
    
    # Neural models setup
    neural_models = [
        ("RNN", SimpleRNNClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE)),
        ("LSTM", LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE)),
        ("GRU", GRUClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, DENSE_DIM, DROPOUT_RATE)),
        ("GRU + Attention", GRUAttentionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM, DENSE_DIM, DROPOUT_RATE))
    ]
    
    gru_attn_weights = None
    
    for name, model in neural_models:
        print(f"\n--- Training {name} ---")
        hist = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=EARLY_STOPPING_PATIENCE,
            device=DEVICE,
            verbose=False
        )
        histories[name] = hist
        
        collect_attn = (name == "GRU + Attention")
        _, _, probs, _, attn_wts = evaluate_epoch(model, test_loader, criterion, DEVICE, collect_attention=collect_attn)
        metrics = compute_classification_metrics(y_test, probs)
        test_probs[name] = probs
        if collect_attn:
            gru_attn_weights = attn_wts
            
        params = model.count_parameters()
        comparison_records.append({
            "model": name,
            "parameters": params,
            "training_time_sec": round(hist["training_time_sec"], 3),
            "epochs_trained": hist["epochs_trained"],
            **metrics
        })
        print(f"{name} -> F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Params: {params} | Time: {hist['training_time_sec']:.3f}s")

    # Save benchmark table
    benchmark_df = pd.DataFrame(comparison_records)
    benchmark_df.to_csv(OUTPUT_DIR / "benchmark.csv", index=False)
    benchmark_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    
    # Training plots (Plots 4-7)
    generate_training_plots(histories, CHARTS_DIR)

    # -------------------------------------------------------------
    # Experiment 2: Attention Dimension Sweep (32, 64, 128)
    # -------------------------------------------------------------
    print("\n" + "="*55)
    print("EXPERIMENT 2: Attention Dimension Sweep (32, 64, 128)")
    print("="*55)
    att_dim_records = []
    for ad in [32, 64, 128]:
        m_ad = GRUAttentionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, attention_dim=ad, dense_dim=DENSE_DIM, dropout_rate=DROPOUT_RATE)
        h_ad = fit_model(m_ad, train_loader, val_loader, epochs=8, lr=LEARNING_RATE, patience=2, device=DEVICE, verbose=False)
        _, _, p_ad, _, _ = evaluate_epoch(m_ad, test_loader, criterion, DEVICE)
        met_ad = compute_classification_metrics(y_test, p_ad)
        att_dim_records.append({
            "experiment": "Exp2_Attention_Dim",
            "attention_dim": ad,
            "parameters": m_ad.count_parameters(),
            "training_time_sec": round(h_ad["training_time_sec"], 3),
            **met_ad
        })
        print(f"Attention Dim {ad} -> F1: {met_ad['f1']:.4f}, Params: {m_ad.count_parameters()}")

    # -------------------------------------------------------------
    # Experiment 3: GRU Hidden Size (32, 64, 128) vs Attention
    # -------------------------------------------------------------
    print("\n" + "="*55)
    print("EXPERIMENT 3: Hidden Size Sweep (GRU vs GRU + Attention)")
    print("="*55)
    hidden_sweep_records = []
    for h_size in [32, 64, 128]:
        # GRU
        m_gru = GRUClassifier(vocab_size, EMBEDDING_DIM, hidden_dim=h_size, dense_dim=DENSE_DIM, dropout_rate=DROPOUT_RATE)
        h_g = fit_model(m_gru, train_loader, val_loader, epochs=8, lr=LEARNING_RATE, patience=2, device=DEVICE, verbose=False)
        _, _, p_g, _, _ = evaluate_epoch(m_gru, test_loader, criterion, DEVICE)
        met_g = compute_classification_metrics(y_test, p_g)
        hidden_sweep_records.append({
            "model": "GRU",
            "hidden_size": h_size,
            "parameters": m_gru.count_parameters(),
            "training_time_sec": round(h_g["training_time_sec"], 3),
            **met_g
        })
        # GRU + Attention
        m_ga = GRUAttentionClassifier(vocab_size, EMBEDDING_DIM, hidden_dim=h_size, attention_dim=64, dense_dim=DENSE_DIM, dropout_rate=DROPOUT_RATE)
        h_ga = fit_model(m_ga, train_loader, val_loader, epochs=8, lr=LEARNING_RATE, patience=2, device=DEVICE, verbose=False)
        _, _, p_ga, _, _ = evaluate_epoch(m_ga, test_loader, criterion, DEVICE)
        met_ga = compute_classification_metrics(y_test, p_ga)
        hidden_sweep_records.append({
            "model": "GRU + Attention",
            "hidden_size": h_size,
            "parameters": m_ga.count_parameters(),
            "training_time_sec": round(h_ga["training_time_sec"], 3),
            **met_ga
        })
        print(f"Hidden {h_size} | GRU F1: {met_g['f1']:.4f} vs GRU+Attn F1: {met_ga['f1']:.4f}")
        
    hidden_sweep_df = pd.DataFrame(hidden_sweep_records)
    hidden_sweep_df.to_csv(OUTPUT_DIR / "gru_attention_benchmark.csv", index=False)

    # -------------------------------------------------------------
    # Experiment 4: Sequence Length Sweep (20, 40, 60, 100)
    # -------------------------------------------------------------
    print("\n" + "="*55)
    print("EXPERIMENT 4: Sequence Length Sweep (20, 40, 60, 100)")
    print("="*55)
    seqlen_records = []
    for sl in [20, 40, 60, 100]:
        X_tr_sl = pad_sequences(train_encoded, max_len=sl)
        X_v_sl = pad_sequences(val_encoded, max_len=sl)
        X_te_sl = pad_sequences(test_encoded, max_len=sl)
        
        tr_l = create_dataloader(X_tr_sl, y_train, BATCH_SIZE, shuffle=True)
        v_l = create_dataloader(X_v_sl, y_val, BATCH_SIZE, shuffle=False)
        te_l = create_dataloader(X_te_sl, y_test, BATCH_SIZE, shuffle=False)
        
        m_sl = GRUAttentionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM, DENSE_DIM, DROPOUT_RATE)
        h_sl = fit_model(m_sl, tr_l, v_l, epochs=8, lr=LEARNING_RATE, patience=2, device=DEVICE, verbose=False)
        _, _, p_sl, _, _ = evaluate_epoch(m_sl, te_l, criterion, DEVICE)
        met_sl = compute_classification_metrics(y_test, p_sl)
        seqlen_records.append({
            "experiment": "Exp4_SeqLength",
            "seq_len": sl,
            "training_time_sec": round(h_sl["training_time_sec"], 3),
            **met_sl
        })
        print(f"SeqLen {sl} -> F1: {met_sl['f1']:.4f}, Time: {h_sl['training_time_sec']:.2f}s")

    # -------------------------------------------------------------
    # Experiment 5: Attention Weight Distribution & Entropy
    # -------------------------------------------------------------
    print("\n" + "="*55)
    print("EXPERIMENT 5: Attention Weight Distribution & Entropy")
    print("="*55)
    
    if gru_attn_weights is not None:
        entropies = compute_attention_entropy(gru_attn_weights)
        dist_summary = summarize_attention_distribution(gru_attn_weights)
        print("Attention Distribution Summary:")
        print(f"  Max Weight:   {dist_summary['max_attention']:.4f}")
        print(f"  Min Weight:   {dist_summary['min_attention']:.4f}")
        print(f"  Mean Weight:  {dist_summary['mean_attention']:.4f}")
        print(f"  Mean Entropy: {dist_summary['entropy']:.4f}")
        
        # Save raw attention weights matrix
        np.savetxt(OUTPUT_DIR / "attention_weights.csv", gru_attn_weights, delimiter=",")
        
        # Aggregate token attention
        token_attn_df = aggregate_token_attention(test_tokenized, gru_attn_weights)
    else:
        entropies = np.zeros(len(y_test))
        token_attn_df = pd.DataFrame()

    # -------------------------------------------------------------
    # Threshold Analysis: Section 33
    # -------------------------------------------------------------
    print("\nRunning Threshold Analysis (0.10 to 0.90)...")
    thresh_gru = analyze_thresholds(y_test, test_probs["GRU"], model_name="GRU")
    thresh_attn = analyze_thresholds(y_test, test_probs["GRU + Attention"], model_name="GRU + Attention")
    thresh_df = pd.concat([thresh_gru, thresh_attn], ignore_index=True)
    thresh_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

    # -------------------------------------------------------------
    # Error Analysis: Section 34
    # -------------------------------------------------------------
    print("Performing message-level error analysis...")
    error_res = perform_error_analysis(
        texts=test_df["text"].tolist(),
        tokenized_texts=test_tokenized,
        y_true=y_test,
        gru_probs=test_probs["GRU"],
        attn_probs=test_probs["GRU + Attention"],
        attn_weights=gru_attn_weights if gru_attn_weights is not None else np.zeros((len(y_test), MAX_SEQ_LENGTH)),
        output_dir=OUTPUT_DIR
    )
    print(f"Errors Summary: GRU wrong & Attn right = {error_res['comparative_summary']['gru_wrong_attn_right_count']}")
    print(f"Errors Summary: GRU right & Attn wrong = {error_res['comparative_summary']['gru_right_attn_wrong_count']}")
    print(f"Errors Summary: Both wrong = {error_res['comparative_summary']['both_wrong_count']}")

    # -------------------------------------------------------------
    # Attention Visualizations & Heatmaps (Sections 23, 24, 35)
    # -------------------------------------------------------------
    print("\nGenerating Attention Heatmaps in output/attention_examples/...")
    if gru_attn_weights is not None:
        generate_attention_example_heatmaps(
            texts=test_df["text"].tolist(),
            tokenized_texts=test_tokenized,
            y_true=y_test,
            y_prob=test_probs["GRU + Attention"],
            attention_weights=gru_attn_weights,
            output_dir=ATTENTION_EXAMPLES_DIR
        )

    # Compute ROC and PR curves data
    roc_data = {m: compute_roc_curve_data(y_test, p) for m, p in test_probs.items()}
    pr_data = {m: compute_pr_curve_data(y_test, p) for m, p in test_probs.items()}
    
    # Generate all benchmark plots (Plots 1-3, 8-23)
    print("Generating Benchmark Visualizations in output/charts/...")
    generate_benchmark_visualizations(
        df_raw=clean_df,
        comparison_df=benchmark_df,
        histories=histories,
        attn_weights=gru_attn_weights,
        token_attn_df=token_attn_df,
        entropies=entropies,
        y_test=y_test,
        model_probs=test_probs,
        roc_data=roc_data,
        pr_data=pr_data,
        thresh_df=thresh_df,
        tokenized_texts=test_tokenized,
        charts_dir=CHARTS_DIR
    )
    
    print("\n" + "="*55)
    print("Attention Benchmark Suite Completed Successfully!")
    print("="*55)
    
    return {
        "benchmark_df": benchmark_df,
        "histories": histories,
        "error_analysis": error_res
    }
