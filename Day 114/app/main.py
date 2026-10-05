"""
Orchestrator for Day 114 Instruction Fine-Tuning & MiniGPT-Chat Experiments:
1. Base Model vs SFT Fine-Tuning Evaluation
2. Overfitting & Epoch Scaling (1 vs 3 vs 5 epochs)
3. Full Fine-Tuning vs LoRA Adaptation
4. Learning Rate Comparison (1e-4 vs 5e-4)
5. Decoding Temperature & Lexical Diversity Analysis
6. Full 50+ Test Prompt Benchmark Scoring
"""
import copy
import time
import json
from pathlib import Path
import pandas as pd
import torch

from app.config import ChatModelConfig, SFTTrainingConfig, GenerationConfig, LoRAConfig
from app.data.dataset import load_jsonl
from app.data.tokenizer import ChatTokenizer
from app.model.minigpt_chat import MiniGPTChat
from app.training.trainer import SFTTrainer
from app.data.formatter import format_chat, extract_assistant_response
from app.evaluation.evaluator import evaluate_model, compare_base_vs_sft, save_evaluation_results
from app.analysis.plots import plot_all_experiments


def run_all_experiments():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DAY 114: INSTRUCTION FINE-TUNING & CHAT MODELS (MiniGPT-Chat)")
    print("=" * 80)

    # 1. Load Data and Tokenizer
    vocab_path = data_dir / "vocab.json"
    tokenizer = ChatTokenizer.load(vocab_path) if vocab_path.exists() else ChatTokenizer()
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")

    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data = load_jsonl(data_dir / "validation.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")
    print(f"Dataset partitions: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Model Configuration
    model_cfg = ChatModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        tie_weights=True
    )

    # -------------------------------------------------------------------------
    # Experiment 1: Base Model vs Full SFT Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n[Experiment 1] Base Model vs Supervised Fine-Tuning (SFT)...")
    torch.manual_seed(42)
    base_model = MiniGPTChat(model_cfg)
    base_eval = evaluate_model(base_model, test_data, tokenizer)
    print(f"  Base Model Instruction Score: {base_eval['overall_score_pct']:.2f}% (Pre-SFT baseline)")

    # SFT Training (3 Epochs default)
    sft_cfg = SFTTrainingConfig(
        num_epochs=3,
        batch_size=16,
        learning_rate=3e-4,
        warmup_steps=10,
        eval_interval=15,
        mask_prompt_loss=True
    )
    sft_model = copy.deepcopy(base_model)
    trainer = SFTTrainer(
        sft_model,
        train_data,
        val_data,
        tokenizer,
        config=sft_cfg,
        output_dir=output_dir,
        experiment_name="sft_standard"
    )
    sft_train_res = trainer.train(verbose=True)

    sft_eval = evaluate_model(sft_model, test_data, tokenizer)
    print(f"  Post-SFT Instruction Score: {sft_eval['overall_score_pct']:.2f}%")
    print(f"  Improvement: +{sft_eval['overall_score_pct'] - base_eval['overall_score_pct']:.2f}%")

    # Save detailed evaluation.json
    save_evaluation_results(sft_eval, output_dir / "evaluation.json")

    # Generate Before vs After Table for sample prompts
    sample_test_prompts = [
        {"instruction": "What is a Python list?", "ground_truth": "A list is an ordered, mutable collection.", "category": "python"},
        {"instruction": "What is overfitting in machine learning?", "ground_truth": "Overfitting occurs when a model learns training data noise.", "category": "data_science"},
        {"instruction": "What is a matrix in linear algebra?", "ground_truth": "A matrix is a rectangular array of numbers.", "category": "mathematics"},
        {"instruction": "What is the largest planet in our solar system?", "ground_truth": "Jupiter is the largest planet.", "category": "general_knowledge"},
        {"instruction": "If all dogs are animals, and Buddy is a dog, is Buddy an animal?", "ground_truth": "Yes, Buddy is an animal.", "category": "reasoning"},
        {"instruction": "Write a Python function to reverse a string.", "ground_truth": "def reverse_string(s): return s[::-1]", "category": "coding"},
    ]
    comparison_results = compare_base_vs_sft(base_model, sft_model, sample_test_prompts, tokenizer)
    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv(metrics_dir / "base_vs_sft_comparison.csv", index=False)
    print(f"  Saved Base vs SFT comparison table ({len(df_comp)} items) to metrics/base_vs_sft_comparison.csv")

    # -------------------------------------------------------------------------
    # Experiment 2: Overfitting & Epoch Scaling (1 vs 3 vs 5 Epochs)
    # -------------------------------------------------------------------------
    print("\n[Experiment 2] Overfitting & Epoch Scaling Analysis (1 vs 3 vs 5 Epochs)...")
    epoch_configs = [1, 3, 5]
    overfitting_records = []

    for ep in epoch_configs:
        print(f"  Training for {ep} epoch(s)...")
        torch.manual_seed(42)
        model_ep = copy.deepcopy(base_model)
        cfg_ep = SFTTrainingConfig(num_epochs=ep, batch_size=16, learning_rate=3e-4, warmup_steps=5, eval_interval=50)
        t_ep = SFTTrainer(model_ep, train_data, val_data, tokenizer, config=cfg_ep, output_dir=output_dir, experiment_name=f"epoch_{ep}")
        res_ep = t_ep.train(verbose=False)
        val_metrics = t_ep.evaluate()
        eval_score = evaluate_model(model_ep, test_data, tokenizer)

        overfitting_records.append({
            "epochs": ep,
            "total_steps": res_ep["total_steps"],
            "final_train_loss": t_ep.history[-1]["train_loss"] if t_ep.history else None,
            "final_val_loss": val_metrics["val_loss"],
            "val_perplexity": val_metrics["val_ppl"],
            "val_token_accuracy": val_metrics["val_acc"],
            "test_instruction_score_pct": eval_score["overall_score_pct"],
            "elapsed_seconds": res_ep["elapsed_seconds"]
        })

    df_overfit = pd.DataFrame(overfitting_records)
    df_overfit.to_csv(metrics_dir / "overfitting_results.csv", index=False)
    print("  Overfitting results summary:")
    print(df_overfit.to_string())

    # -------------------------------------------------------------------------
    # Experiment 3: Full Fine-Tuning vs LoRA Adaptation
    # -------------------------------------------------------------------------
    print("\n[Experiment 3] Full Fine-Tuning vs Low-Rank Adaptation (LoRA)...")
    torch.manual_seed(42)
    lora_model = copy.deepcopy(base_model)
    lora_stats = lora_model.apply_lora(r=8, lora_alpha=16.0, lora_dropout=0.05, target_modules=["c_attn", "c_proj"])

    print(f"  Full SFT Trainable Params: {base_model.count_parameters()['trainable']:,} (100.0%)")
    print(f"  LoRA Trainable Params: {lora_stats['trainable']:,} ({lora_stats['trainable_percent']}%)")

    lora_cfg = SFTTrainingConfig(num_epochs=3, batch_size=16, learning_rate=5e-4, warmup_steps=10, eval_interval=15)
    lora_trainer = SFTTrainer(lora_model, train_data, val_data, tokenizer, config=lora_cfg, output_dir=output_dir, experiment_name="lora_r8")
    lora_res = lora_trainer.train(verbose=False)
    lora_val = lora_trainer.evaluate()
    lora_eval = evaluate_model(lora_model, test_data, tokenizer)

    lora_comparison = [
        {
            "tuning_mode": "Full Fine-Tuning",
            "total_parameters": base_model.count_parameters()["total"],
            "trainable_parameters": base_model.count_parameters()["trainable"],
            "trainable_percent": 100.0,
            "final_val_loss": sft_train_res["best_val_loss"],
            "test_instruction_score_pct": sft_eval["overall_score_pct"],
            "memory_efficiency": "Baseline (1.0x)",
            "adapter_size_kb": 0.0
        },
        {
            "tuning_mode": "LoRA (r=8, alpha=16)",
            "total_parameters": lora_stats["total"],
            "trainable_parameters": lora_stats["trainable"],
            "trainable_percent": lora_stats["trainable_percent"],
            "final_val_loss": lora_val["val_loss"],
            "test_instruction_score_pct": lora_eval["overall_score_pct"],
            "memory_efficiency": f"{round(base_model.count_parameters()['trainable'] / lora_stats['trainable'], 1)}x parameter reduction",
            "adapter_size_kb": round((lora_stats["trainable"] * 4) / 1024, 2)
        }
    ]
    df_lora = pd.DataFrame(lora_comparison)
    df_lora.to_csv(metrics_dir / "lora_comparison.csv", index=False)
    print("  LoRA comparison summary:")
    print(df_lora.to_string())

    # -------------------------------------------------------------------------
    # Experiment 4: Learning Rate Dynamics (1e-4 vs 5e-4)
    # -------------------------------------------------------------------------
    print("\n[Experiment 4] Learning Rate Dynamics (1e-4 vs 5e-4)...")
    lrs = [1e-4, 5e-4]
    lr_records = []

    for lr in lrs:
        torch.manual_seed(42)
        m_lr = copy.deepcopy(base_model)
        cfg_lr = SFTTrainingConfig(num_epochs=3, batch_size=16, learning_rate=lr, warmup_steps=10, eval_interval=50)
        t_lr = SFTTrainer(m_lr, train_data, val_data, tokenizer, config=cfg_lr, output_dir=output_dir, experiment_name=f"lr_{lr}")
        res_lr = t_lr.train(verbose=False)
        val_lr = t_lr.evaluate()
        score_lr = evaluate_model(m_lr, test_data, tokenizer)

        lr_records.append({
            "learning_rate": lr,
            "final_train_loss": t_lr.history[-1]["train_loss"] if t_lr.history else None,
            "final_val_loss": val_lr["val_loss"],
            "final_val_ppl": val_lr["val_ppl"],
            "val_token_acc": val_lr["val_acc"],
            "instruction_score_pct": score_lr["overall_score_pct"]
        })

    df_lr = pd.DataFrame(lr_records)
    df_lr.to_csv(metrics_dir / "learning_rate_results.csv", index=False)
    print("  Learning rate comparison summary:")
    print(df_lr.to_string())

    # -------------------------------------------------------------------------
    # Experiment 5: Decoding Temperature & Sampling Diversity
    # -------------------------------------------------------------------------
    print("\n[Experiment 5] Decoding Temperature & Sampling Diversity...")
    test_prompts_temp = [
        "Explain Python lists.",
        "What is overfitting in machine learning?",
        "Write a Python function to reverse a string."
    ]
    temperatures = [0.2, 0.7, 1.2]
    temp_records = []

    for temp in temperatures:
        gen_cfg = GenerationConfig(temperature=temp, max_new_tokens=48, top_k=40, top_p=0.9, do_sample=True)
        all_generated_tokens = []
        sample_outputs = []

        for p in test_prompts_temp:
            p_msgs = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": p}
            ]
            f_text = format_chat(p_msgs, add_generation_prompt=True)
            p_ids = torch.tensor([tokenizer.encode(f_text)], dtype=torch.long)
            with torch.no_grad():
                out = sft_model.generate(p_ids, max_new_tokens=48, temperature=temp, do_sample=True)
            rep = extract_assistant_response(tokenizer.decode(out[0].tolist()))
            sample_outputs.append(f"Q: '{p}' -> A: '{rep[:50]}...'")
            words = rep.lower().split()
            all_generated_tokens.extend(words)

        unique_tokens = len(set(all_generated_tokens))
        total_tokens = len(all_generated_tokens)
        ttr = round(unique_tokens / max(1, total_tokens), 4)

        temp_records.append({
            "temperature": temp,
            "type_token_ratio (diversity)": ttr,
            "total_tokens_generated": total_tokens,
            "unique_tokens_generated": unique_tokens,
            "sample_snippet": sample_outputs[0]
        })

    df_temp = pd.DataFrame(temp_records)
    df_temp.to_csv(metrics_dir / "temperature_results.csv", index=False)
    print("  Temperature diversity summary:")
    print(df_temp[["temperature", "type_token_ratio (diversity)", "unique_tokens_generated"]].to_string())

    plots_dir = output_dir / "plots"
    plot_all_experiments(metrics_dir, plots_dir)
    print(f"\n[Visualizations] All 6 publication charts generated in: {plots_dir}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! Metrics saved to Day 114/outputs/metrics/")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()
