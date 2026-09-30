import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import pytest
import torch
import torch.nn as nn
from app.config import ModelConfig, TrainingConfig
from app.model.minigpt import ScalableMiniGPT
from app.training.scheduler import configure_scheduler
from app.training.checkpoint import save_checkpoint, load_checkpoint
from app.training.logger import TrainingCSVLogger
from app.training.trainer import ScalingTrainer, sample_batch


class TestLearningRateSchedulers:
    def test_constant_scheduler_returns_base_lr(self):
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        scheduler = configure_scheduler(optimizer, "constant", warmup_steps=10, max_steps=100)
        
        for _ in range(20):
            optimizer.step()
            scheduler.step()
            assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3, abs=1e-7)

    def test_linear_scheduler_warmup_phase(self):
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        scheduler = configure_scheduler(optimizer, "linear", warmup_steps=10, max_steps=100, min_lr_ratio=0.0)
        
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3, abs=1e-5)

    def test_linear_scheduler_decay_phase(self):
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        scheduler = configure_scheduler(optimizer, "linear", warmup_steps=0, max_steps=100, min_lr_ratio=0.1)
        
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-4, abs=1e-5)
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4, abs=1e-5)

    def test_cosine_scheduler_warmup_phase(self):
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        scheduler = configure_scheduler(optimizer, "cosine", warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
        
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4, abs=1e-4)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3, abs=1e-5)

    def test_cosine_scheduler_cosine_decay(self):
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        scheduler = configure_scheduler(optimizer, "cosine", warmup_steps=0, max_steps=100, min_lr_ratio=0.0)
        
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4, abs=1e-4)
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_scheduler_raises_value_error(self):
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            configure_scheduler(optimizer, "exponential", warmup_steps=10, max_steps=100)


class TestCheckpointer:
    def test_save_and_load_checkpoint_roundtrip(self, tiny_model, tiny_config):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "subfolder" / "ckpt_step_10.pt"
            save_checkpoint(
                filepath=ckpt_path,
                model=tiny_model,
                optimizer=optimizer,
                step=10,
                config=tiny_config.__dict__,
                metrics={"val_loss": 2.45}
            )
            assert ckpt_path.exists()

            loaded_model = ScalableMiniGPT(tiny_config)
            loaded_opt = torch.optim.AdamW(loaded_model.parameters(), lr=1e-3)
            meta = load_checkpoint(ckpt_path, loaded_model, loaded_opt)

            assert meta["step"] == 10
            assert meta["metrics"]["val_loss"] == pytest.approx(2.45, abs=1e-5)

            for p1, p2 in zip(tiny_model.parameters(), loaded_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_load_nonexistent_checkpoint_raises(self, tiny_model):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent_checkpoint_file.pt", tiny_model)

    def test_save_checkpoint_without_optimizer(self, tiny_model, tiny_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt_no_opt.pt"
            save_checkpoint(
                filepath=ckpt_path,
                model=tiny_model,
                optimizer=None,
                step=5,
                config=tiny_config.__dict__,
                metrics={"val_loss": 3.10}
            )
            loaded_model = ScalableMiniGPT(tiny_config)
            meta = load_checkpoint(ckpt_path, loaded_model, optimizer=None)
            assert meta["step"] == 5
            assert meta["metrics"]["val_loss"] == pytest.approx(3.10, abs=1e-5)


class TestTrainingCSVLogger:
    def test_logger_creates_file_and_writes_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_log.csv"
            logger = TrainingCSVLogger(log_path)
            assert log_path.exists()
            content = log_path.read_text(encoding="utf-8")
            assert "step,loss,val_loss,val_ppl,learning_rate" in content

    def test_logger_appends_step_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "metrics.csv"
            logger = TrainingCSVLogger(log_path)
            logger.log(step=1, loss=3.5, val_loss=3.4, val_ppl=30.0, learning_rate=1e-4, tokens_seen=128, tokens_per_sec=1500.0, memory_mb=120.0, elapsed_time=0.1)
            logger.log(step=2, loss=3.1, val_loss=3.0, val_ppl=20.0, learning_rate=2e-4, tokens_seen=256, tokens_per_sec=1600.0, memory_mb=122.0, elapsed_time=0.2)

            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 3
            assert "3.5" in lines[1]
            assert "3.1" in lines[2]

            df = logger.to_dataframe()
            assert len(df) == 2
            assert df.iloc[0]["loss"] == 3.5


class TestScalingTrainer:
    def test_sample_batch_shapes(self):
        data = torch.arange(100)
        xb, yb = sample_batch(data, batch_size=4, context_length=16, device=torch.device("cpu"))
        assert xb.shape == (4, 16)
        assert yb.shape == (4, 16)
        # Check target is right-shifted by 1
        assert torch.all(yb[:, :-1] == xb[:, 1:])

    def test_sample_batch_too_short_raises(self):
        data = torch.arange(10)
        with pytest.raises(ValueError, match="must be strictly greater"):
            sample_batch(data, batch_size=2, context_length=16, device=torch.device("cpu"))

    def test_trainer_initialization(self, tiny_model, tiny_config):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        train_cfg = TrainingConfig(
            max_steps=10,
            batch_size=4,
            gradient_accumulation_steps=2,
            eval_interval=5
        )
        trainer = ScalingTrainer(
            model=tiny_model,
            optimizer=optimizer,
            config=train_cfg,
            device=torch.device("cpu")
        )
        assert trainer.device == torch.device("cpu")
        assert trainer.config.gradient_accumulation_steps == 2
        assert trainer.get_memory_mb() > 0

    def test_trainer_estimate_loss_leaves_model_in_train_mode(self, tiny_model, tiny_config):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        train_cfg = TrainingConfig(max_steps=5, eval_iters=2)
        trainer = ScalingTrainer(
            model=tiny_model,
            optimizer=optimizer,
            config=train_cfg,
            device=torch.device("cpu")
        )
        tokens = torch.randint(0, 32, (100,))
        tiny_model.train()
        eval_metrics = trainer.estimate_loss(tokens, context_length=16, eval_iters=2, batch_size=2)
        assert "loss" in eval_metrics
        assert "perplexity" in eval_metrics
        assert isinstance(eval_metrics["loss"], float)
        assert isinstance(eval_metrics["perplexity"], float)

    def test_trainer_reproducibility_with_fixed_seed(self, tiny_config):
        train_cfg = TrainingConfig(max_steps=5, batch_size=4, gradient_accumulation_steps=1, seed=42)
        tokens = torch.randint(0, 32, (200,))

        torch.manual_seed(42)
        m1 = ScalableMiniGPT(tiny_config)
        opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
        t1 = ScalingTrainer(m1, opt1, config=train_cfg, device=torch.device("cpu"))
        res1 = t1.train(tokens, tokens, context_length=16, verbose=False)

        torch.manual_seed(42)
        m2 = ScalableMiniGPT(tiny_config)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
        t2 = ScalingTrainer(m2, opt2, config=train_cfg, device=torch.device("cpu"))
        res2 = t2.train(tokens, tokens, context_length=16, verbose=False)

        assert res1["final_train_loss"] == pytest.approx(res2["final_train_loss"], abs=1e-4)
