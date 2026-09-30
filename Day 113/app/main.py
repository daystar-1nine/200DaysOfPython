"""
Main Orchestrator for Day 113: LLM Training Simulator & MiniGPT Scaling Lab.
Executes all 6 empirical experiments:
1. Model Architecture Scaling (Tiny vs Small vs Medium)
2. Gradient Accumulation (Batch 32, Accum 1 vs Accum 4)
3. Learning Rate Schedules (Constant vs Warmup + Cosine Decay)
4. Data Quality & Duplication (Clean vs 4x Duplicated)
5. Data Leakage & Contamination (Clean Split vs 50% Leaked)
6. Precision & Memory Scaling (FP32 vs BF16 vs INT8 vs INT4)
"""
import random
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from app.config import SCALING_MODELS, ModelConfig, TrainingConfig
from app.data import (
    load_raw_text,
    split_into_documents,
    normalize_text,
    filter_documents,
    exact_deduplicate,
    near_deduplicate,
    train_val_split,
    detect_leakage,
    inject_leakage,
    build_character_vocab
)
from app.model import ScalableMiniGPT
from app.training import ScalingTrainer, TrainingCSVLogger, configure_scheduler
from app.analysis.params import estimate_parameters, parameter_memory, estimate_training_memory
from app.analysis.compute import estimate_training_flops, tokens_per_second


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_all_experiments():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "input.txt"
    outputs_dir = base_dir / "outputs"
    metrics_dir = outputs_dir / "metrics"
    checkpoints_dir = outputs_dir / "checkpoints"
    plots_dir = outputs_dir / "plots"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DAY 113: LLM TRAINING SIMULATOR & MINIGPT SCALING LAB")
    print("=" * 80)

    # ---------------------------------------------------------
    # STAGE 1: Data Pipeline Execution
    # ---------------------------------------------------------
    print("\n[Stage 1] Executing LLM Data Pipeline (Load -> Clean -> Dedup -> Leak-Free Split)...")
    raw_text = load_raw_text(data_path)
    print(f"  Raw corpus characters: {len(raw_text):,}")

    raw_docs = split_into_documents(raw_text, delimiter="\n\n")
    print(f"  Extracted document chunks: {len(raw_docs)}")

    clean_docs, rejected = filter_documents(raw_docs, min_quality=0.40)
    print(f"  Quality-filtered documents: {len(clean_docs)} retained ({len(rejected)} rejected)")

    dedup_docs, dupes_removed = exact_deduplicate(clean_docs)
    print(f"  Exact deduplication: {len(dedup_docs)} documents ({dupes_removed} duplicates purged)")

    train_docs, val_docs = train_val_split(dedup_docs, val_ratio=0.15, seed=42)
    leakage_audit = detect_leakage(train_docs, val_docs)
    print(f"  Split: {len(train_docs)} train docs, {len(val_docs)} val docs")
    print(f"  Leakage Audit: Exact leaks = {leakage_audit['exact_leaked_documents']}, N-gram overlap = {leakage_audit['ngram_overlap_ratio']:.4f}")

    train_text = "\n\n".join(train_docs)
    val_text = "\n\n".join(val_docs)

    full_corpus = train_text + "\n\n" + val_text
    vocab, char2idx, idx2char = build_character_vocab(full_corpus)
    vocab_size = len(vocab)
    print(f"  Vocabulary size: {vocab_size} unique characters")

    train_tokens = torch.tensor([char2idx.get(c, 0) for c in train_text], dtype=torch.long)
    val_tokens = torch.tensor([char2idx.get(c, 0) for c in val_text], dtype=torch.long)
    print(f"  Encoded: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")

    # ---------------------------------------------------------
    # EXPERIMENT 1: Model Architecture Scaling (Tiny, Small, Medium)
    # ---------------------------------------------------------
    print("\n[Experiment 1] Model Architecture Scaling (Tiny vs Small vs Medium)...")
    scaling_records = []
    trained_models = {}

    model_tiers = [
        ("tiny", SCALING_MODELS["tiny"], 100),
        ("small", SCALING_MODELS["small"], 100),
        ("medium", SCALING_MODELS["medium"], 80)
    ]

    for key, cfg, max_steps in model_tiers:
        set_seed(42)
        cfg.vocab_size = vocab_size
        model = ScalableMiniGPT(cfg)
        param_counts = model.count_parameters()
        total_params = param_counts["total"]

        exp_logger = TrainingCSVLogger(metrics_dir / f"training_{key}.csv")
        t_cfg = TrainingConfig(
            max_steps=max_steps,
            batch_size=32,
            learning_rate=5e-4 if key == "tiny" else (4e-4 if key == "small" else 3e-4),
            warmup_steps=15,
            eval_interval=max(1, max_steps // 4),
            eval_iters=8
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg.learning_rate, weight_decay=t_cfg.weight_decay)
        trainer = ScalingTrainer(
            model=model,
            optimizer=optimizer,
            config=t_cfg,
            logger=exp_logger
        )

        print(f"  Training {cfg.name} ({total_params:,} parameters) for {max_steps} steps...")
        ckpt_dir = checkpoints_dir / key
        run_res = trainer.train(train_tokens, val_tokens, context_length=cfg.context_length, checkpoint_dir=ckpt_dir, verbose=True)

        flops_res = estimate_training_flops(total_params, run_res["total_tokens_seen"])

        record = {
            "model_name": cfg.name,
            "parameters": total_params,
            "embed_dim": cfg.embed_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "train_loss": run_res["final_train_loss"],
            "val_loss": run_res["final_val_loss"],
            "val_perplexity": run_res["final_val_ppl"],
            "tokens_per_sec": run_res["average_tokens_per_second"],
            "total_time_seconds": run_res["total_time_seconds"],
            "training_flops": flops_res["total_training_flops"],
            "memory_mb": run_res["peak_memory_mb"]
        }
        scaling_records.append(record)
        trained_models[key] = model

    df_scaling = pd.DataFrame(scaling_records)
    df_scaling.to_csv(metrics_dir / "scaling_results.csv", index=False)
    df_scaling.to_csv(outputs_dir / "scaling_results.csv", index=False)
    print("\nScaling Results Summary:")
    print(df_scaling[["model_name", "parameters", "train_loss", "val_loss", "val_perplexity", "tokens_per_sec", "memory_mb"]].to_string(index=False))

    # ---------------------------------------------------------
    # EXPERIMENT 2: Gradient Accumulation (Effective Batch Scaling)
    # ---------------------------------------------------------
    print("\n[Experiment 2] Gradient Accumulation Experiment (Batch 32: Accum 1 vs Micro-Batch 8, Accum 4)...")
    accum_records = []
    for accum_steps in [1, 4]:
        set_seed(42)
        cfg_tiny = SCALING_MODELS["tiny"]
        cfg_tiny.vocab_size = vocab_size
        model = ScalableMiniGPT(cfg_tiny)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        t_cfg = TrainingConfig(
            max_steps=60,
            batch_size=32,
            gradient_accumulation_steps=accum_steps,
            eval_interval=30
        )
        logger = TrainingCSVLogger(metrics_dir / f"accum_{accum_steps}.csv")
        trainer = ScalingTrainer(model=model, optimizer=optimizer, config=t_cfg, logger=logger)

        res = trainer.train(train_tokens, val_tokens, context_length=cfg_tiny.context_length, verbose=False)
        accum_records.append({
            "gradient_accumulation_steps": accum_steps,
            "effective_batch_size": 32,
            "micro_batch_size": 32 // accum_steps,
            "train_loss": res["final_train_loss"],
            "val_loss": res["final_val_loss"],
            "val_perplexity": res["final_val_ppl"],
            "tokens_per_sec": res["average_tokens_per_second"],
            "memory_mb": res["peak_memory_mb"]
        })
        print(f"  Accumulation Steps {accum_steps} | Micro-Batch {32//accum_steps} -> Final Loss: {res['final_train_loss']} | Val PPL: {res['final_val_ppl']} | TPS: {res['average_tokens_per_second']}")

    df_accum = pd.DataFrame(accum_records)
    df_accum.to_csv(metrics_dir / "gradient_accumulation_results.csv", index=False)

    # ---------------------------------------------------------
    # EXPERIMENT 3: Learning Rate Schedules (Constant vs Cosine Warmup)
    # ---------------------------------------------------------
    print("\n[Experiment 3] Learning Rate Schedule Comparison (Constant vs Cosine Warmup)...")
    schedule_records = []
    for sched_type in ["constant", "cosine"]:
        set_seed(42)
        cfg_tiny = SCALING_MODELS["tiny"]
        cfg_tiny.vocab_size = vocab_size
        model = ScalableMiniGPT(cfg_tiny)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        t_cfg = TrainingConfig(
            max_steps=80,
            batch_size=32,
            scheduler_type=sched_type,
            warmup_steps=15,
            eval_interval=20
        )
        logger = TrainingCSVLogger(metrics_dir / f"sched_{sched_type}.csv")
        trainer = ScalingTrainer(model=model, optimizer=optimizer, config=t_cfg, logger=logger)

        res = trainer.train(train_tokens, val_tokens, context_length=cfg_tiny.context_length, verbose=False)
        schedule_records.append({
            "scheduler_type": sched_type,
            "train_loss": res["final_train_loss"],
            "val_loss": res["final_val_loss"],
            "val_perplexity": res["final_val_ppl"],
            "tokens_per_sec": res["average_tokens_per_second"]
        })
        print(f"  Scheduler: {sched_type:<8} -> Final Train Loss: {res['final_train_loss']} | Val Loss: {res['final_val_loss']} | Val PPL: {res['final_val_ppl']}")

    df_sched = pd.DataFrame(schedule_records)
    df_sched.to_csv(metrics_dir / "lr_schedule_results.csv", index=False)

    # ---------------------------------------------------------
    # EXPERIMENT 4: Data Quality & Duplication Impact
    # ---------------------------------------------------------
    print("\n[Experiment 4] Data Quality & Duplication Experiment...")
    # Dataset A: Clean train text
    # Dataset B: 4x duplicated train text (simulating duplicated crawl data)
    duplicated_train_tokens = torch.cat([train_tokens] * 4)

    duplication_records = []
    for mode, d_train in [("Clean (Unique)", train_tokens), ("Duplicated (4x Duplicate)", duplicated_train_tokens)]:
        set_seed(42)
        cfg_tiny = SCALING_MODELS["tiny"]
        cfg_tiny.vocab_size = vocab_size
        model = ScalableMiniGPT(cfg_tiny)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        t_cfg = TrainingConfig(max_steps=60, batch_size=32, eval_interval=30)
        trainer = ScalingTrainer(model=model, optimizer=optimizer, config=t_cfg)
        res = trainer.train(d_train, val_tokens, context_length=cfg_tiny.context_length, verbose=False)
        duplication_records.append({
            "dataset_condition": mode,
            "train_tokens_pool": len(d_train),
            "final_train_loss": res["final_train_loss"],
            "final_val_loss": res["final_val_loss"],
            "final_val_perplexity": res["final_val_ppl"]
        })
        print(f"  {mode:<24} -> Train Loss: {res['final_train_loss']} | Val Loss: {res['final_val_loss']} | Val PPL: {res['final_val_ppl']}")

    df_dupe = pd.DataFrame(duplication_records)
    df_dupe.to_csv(metrics_dir / "data_quality_results.csv", index=False)

    # ---------------------------------------------------------
    # EXPERIMENT 5: Data Leakage & Evaluation Contamination
    # ---------------------------------------------------------
    print("\n[Experiment 5] Data Leakage Experiment (Clean vs 50% Contaminated)...")
    leaked_train_docs = inject_leakage(train_docs, val_docs, leakage_fraction=0.50)
    leaked_audit = detect_leakage(leaked_train_docs, val_docs)
    leaked_train_text = "\n\n".join(leaked_train_docs)
    leaked_train_tokens = torch.tensor([char2idx.get(c, 0) for c in leaked_train_text], dtype=torch.long)

    leakage_records = []
    for mode, d_train, audit in [
        ("Clean (0% Leakage)", train_tokens, leakage_audit),
        ("Contaminated (50% Leakage)", leaked_train_tokens, leaked_audit)
    ]:
        set_seed(42)
        cfg_tiny = SCALING_MODELS["tiny"]
        cfg_tiny.vocab_size = vocab_size
        model = ScalableMiniGPT(cfg_tiny)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        t_cfg = TrainingConfig(max_steps=60, batch_size=32, eval_interval=30)
        trainer = ScalingTrainer(model=model, optimizer=optimizer, config=t_cfg)
        res = trainer.train(d_train, val_tokens, context_length=cfg_tiny.context_length, verbose=False)

        leakage_records.append({
            "condition": mode,
            "exact_leaks": audit["exact_leaked_documents"],
            "ngram_leakage_ratio": audit["ngram_overlap_ratio"],
            "apparent_val_loss": res["final_val_loss"],
            "apparent_val_perplexity": res["final_val_ppl"]
        })
        print(f"  {mode:<26} -> Apparent Val Loss: {res['final_val_loss']} | Apparent PPL: {res['final_val_ppl']} (Exact Leaks: {audit['exact_leaked_documents']})")

    df_leak = pd.DataFrame(leakage_records)
    df_leak.to_csv(metrics_dir / "data_leakage_results.csv", index=False)

    # ---------------------------------------------------------
    # EXPERIMENT 6: Precision & Memory Scaling Matrix
    # ---------------------------------------------------------
    print("\n[Experiment 6] Precision & Memory Footprint Scaling Matrix...")
    target_models = {
        "Tiny (0.8M)": 809_728,
        "Small (4.7M)": 4_771_840,
        "Medium (14.2M)": 14_245_632,
        "GPT-2 (124M)": 124_440_000,
        "LLaMA (7B)": 7_000_000_000
    }
    precisions = ["fp32", "fp16", "int8", "int4"]
    mem_records = []
    for m_name, p_count in target_models.items():
        row = {"model_name": m_name, "parameters": p_count}
        for p in precisions:
            row[f"param_memory_mb_{p}"] = parameter_memory(p_count, precision=p)["megabytes"]
            row[f"param_memory_gb_{p}"] = parameter_memory(p_count, precision=p)["gigabytes"]
        # Training memory (AdamW FP32 vs BF16)
        train_mem_fp32 = estimate_training_memory(p_count, precision="fp32")["total_training_memory_gb"]
        train_mem_bf16 = estimate_training_memory(p_count, precision="bf16")["total_training_memory_gb"]
        row["training_memory_fp32_gb"] = train_mem_fp32
        row["training_memory_bf16_gb"] = train_mem_bf16
        mem_records.append(row)

    df_mem = pd.DataFrame(mem_records)
    df_mem.to_csv(metrics_dir / "precision_memory_results.csv", index=False)
    print("  Precision and memory matrix generated successfully.")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! Metrics saved to Day 113/outputs/metrics/")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()
