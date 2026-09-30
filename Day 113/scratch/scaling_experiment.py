"""
Standalone script to execute scaling experiments across Tiny, Small, and Medium models.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from app.config import SCALING_MODELS, TrainingConfig
from app.data import load_raw_text, build_character_vocab
from app.model import ScalableMiniGPT
from app.training import ScalingTrainer


def run_quick_scaling_demo():
    print("=" * 70)
    print("RUNNING SCALING EXPERIMENT DEMO (TINY vs SMALL vs MEDIUM)")
    print("=" * 70)

    corpus_path = Path(__file__).resolve().parent.parent / "data" / "input.txt"
    text = load_raw_text(corpus_path)
    vocab, char2idx, idx2char = build_character_vocab(text)
    encoded = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)

    split = int(len(encoded) * 0.9)
    train_data = encoded[:split]
    val_data = encoded[split:]

    cfg_train = TrainingConfig(max_steps=50, eval_interval=25, batch_size=16)

    for name in ["tiny", "small"]:
        cfg_model = SCALING_MODELS[name]
        cfg_model.vocab_size = len(vocab)
        model = ScalableMiniGPT(cfg_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        trainer = ScalingTrainer(model=model, optimizer=optimizer, config=cfg_train)

        print(f"\nTraining {cfg_model.name} ({model.count_parameters()['total']:,} params)...")
        res = trainer.train(train_data, val_data, verbose=False)
        print(f"  {cfg_model.name} Finished | Train Loss: {res['final_train_loss']} | Val Loss: {res['final_val_loss']} | PPL: {res['final_val_ppl']} | TPS: {res['average_tokens_per_second']}")

    print("=" * 70)


if __name__ == "__main__":
    run_quick_scaling_demo()
