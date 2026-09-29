"""
Main execution pipeline for Day 112: MiniGPT & Autoregressive Transformers.
"""
from pathlib import Path
import torch

from app.config import config
from app.data.dataset import load_corpus, prepare_data_tensors
from app.tokenizer.char_tokenizer import CharacterTokenizer
from app.model.gpt import MiniGPT
from app.training.optimizer import configure_adamw_optimizer
from app.training.trainer import MiniGPTTrainer
from app.training.checkpoint import save_checkpoint
from app.generation.sampling import generate_with_strategy
from app.evaluation.plots import (
    plot_training_history,
    run_temperature_experiment,
    run_strategy_comparison
)


def run_pipeline() -> None:
    print("=" * 70)
    print("DAY 112: MINIGPT & AUTOREGRESSIVE TRANSFORMERS PIPELINE")
    print("=" * 70)

    # 1. Load Corpus
    print("\n[Step 1] Loading raw corpus...")
    text = load_corpus(config.INPUT_FILE)
    print(f"Corpus loaded: {len(text):,} characters")

    # 2. Tokenizer
    print("\n[Step 2] Initializing Character Tokenizer...")
    tokenizer = CharacterTokenizer.from_text(text)
    vocab_path = config.OUTPUTS_DIR / "vocab.json"
    tokenizer.save(vocab_path)
    print(f"Vocabulary Size: {tokenizer.vocab_size} unique characters | Saved to: {vocab_path}")

    # 3. Data Tensors
    print("\n[Step 3] Preparing train / validation splits...")
    train_data, val_data = prepare_data_tensors(text, tokenizer, val_ratio=config.VAL_RATIO)
    print(f"Train Tokens: {len(train_data):,} | Validation Tokens: {len(val_data):,}")

    # 4. Initialize MiniGPT Model
    print("\n[Step 4] Initializing MiniGPT Architecture...")
    device = torch.device(config.DEVICE)
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=config.CONTEXT_LENGTH,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    params = model.count_parameters()
    print(f"Total Parameters: {params['total']:,} | Context Window: {config.CONTEXT_LENGTH} | Embed Dim: {config.EMBED_DIM} | Layers: {config.NUM_LAYERS}")

    # 5. Configure Optimizer
    print("\n[Step 5] Configuring AdamW Optimizer with weight decay decoupling...")
    optimizer = configure_adamw_optimizer(
        model,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 6. Training
    print("\n[Step 6] Training MiniGPT with teacher forcing...")
    trainer = MiniGPTTrainer(
        model=model,
        optimizer=optimizer,
        context_length=config.CONTEXT_LENGTH,
        batch_size=config.BATCH_SIZE,
        grad_clip=config.GRAD_CLIP,
        device=device
    )

    metrics_csv = config.OUTPUTS_DIR / "metrics.csv"
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        max_iters=config.MAX_ITERS,
        eval_interval=config.EVAL_INTERVAL,
        eval_iters=config.EVAL_ITERS,
        output_csv=metrics_csv,
        verbose=True
    )

    # 7. Save Checkpoint
    checkpoint_path = config.OUTPUTS_DIR / "minigpt_final.pt"
    save_checkpoint(
        model=model,
        filepath=checkpoint_path,
        optimizer=optimizer,
        step=config.MAX_ITERS,
        loss=history["val_loss"][-1] if history["val_loss"] else 0.0,
        metadata={"vocab_size": tokenizer.vocab_size, "context_length": config.CONTEXT_LENGTH}
    )
    print(f"Final model checkpoint saved to: {checkpoint_path}")

    # 8. Plot Training History
    print("\n[Step 7] Generating Loss & Perplexity Trajectory Charts...")
    charts_path = config.CHARTS_DIR / "loss_curves.png"
    plot_training_history(history, charts_path)

    # 9. Temperature Sensitivity Experiment
    print("\n[Step 8] Running Temperature Sensitivity Analysis...")
    temp_report = config.OUTPUTS_DIR / "temperature_comparison.md"
    run_temperature_experiment(
        model=model,
        tokenizer=tokenizer,
        prompt="First Citizen:\n",
        temperatures=(0.3, 0.7, 1.0, 1.3),
        output_md=temp_report
    )

    # 10. Decoding Strategy Benchmark
    print("\n[Step 9] Running Decoding Strategy Benchmark (Greedy vs Sampling vs Top-k vs Top-p)...")
    strategy_report = config.OUTPUTS_DIR / "generation_comparison.md"
    run_strategy_comparison(
        model=model,
        tokenizer=tokenizer,
        prompt="MENENIUS:\n",
        output_md=strategy_report
    )

    # 11. Multi-Prompt Generation
    print("\n[Step 10] Generating samples across canonical prompts...")
    sample_txt_path = config.OUTPUTS_DIR / "sample_generations.txt"
    sample_lines = ["=" * 60 + "\nMINIGPT SAMPLE GENERATIONS\n" + "=" * 60 + "\n\n"]

    for prompt in config.PROMPTS:
        continuation = generate_with_strategy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=120,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            device=device
        )
        sample_lines.append(f"--- Prompt: {repr(prompt)} ---\n{continuation}\n\n")

    with open(sample_txt_path, "w", encoding="utf-8") as f:
        f.writelines(sample_lines)
    print(f"Sample generations saved to: {sample_txt_path}")

    print("\n" + "=" * 70)
    print("DAY 112 PIPELINE COMPLETE! All artifacts generated in Day 112/outputs/")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
