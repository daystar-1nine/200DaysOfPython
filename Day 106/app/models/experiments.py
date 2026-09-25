"""
Experiment configurations and factory for Day 106: RNNs & Sequential Text Learning.
Defines parameters for Experiments A, B, C, and D.
"""

from typing import Any, Dict
from .rnn import SimpleRNNClassifier


class ExperimentConfigs:
    """Standardized configurations for controlled experiments."""

    @staticmethod
    def get_experiment_a(vocab_size: int) -> Dict[str, Any]:
        """Experiment A: Compact model (Embedding 32, RNN 32, Dropout 0.2)."""
        return {
            "name": "Exp_A_Compact_32",
            "vocab_size": vocab_size,
            "embedding_dim": 32,
            "hidden_units": 32,
            "dense_units": 16,
            "dropout_1": 0.2,
            "dropout_2": 0.1,
            "seq_length": 40,
        }

    @staticmethod
    def get_experiment_b(vocab_size: int) -> Dict[str, Any]:
        """Experiment B: Baseline model (Embedding 64, RNN 64, Dropout 0.3)."""
        return {
            "name": "Exp_B_Baseline_64",
            "vocab_size": vocab_size,
            "embedding_dim": 64,
            "hidden_units": 64,
            "dense_units": 32,
            "dropout_1": 0.3,
            "dropout_2": 0.2,
            "seq_length": 40,
        }

    @staticmethod
    def get_experiment_c(vocab_size: int) -> Dict[str, Any]:
        """Experiment C: High capacity (Embedding 128, RNN 128, Dropout 0.3)."""
        return {
            "name": "Exp_C_HighCap_128",
            "vocab_size": vocab_size,
            "embedding_dim": 128,
            "hidden_units": 128,
            "dense_units": 64,
            "dropout_1": 0.3,
            "dropout_2": 0.2,
            "seq_length": 40,
        }

    @staticmethod
    def get_experiment_d(vocab_size: int, seq_length: int) -> Dict[str, Any]:
        """Experiment D: Sequence length variation (T in [20, 40, 60, 100])."""
        return {
            "name": f"Exp_D_SeqLen_{seq_length}",
            "vocab_size": vocab_size,
            "embedding_dim": 64,
            "hidden_units": 64,
            "dense_units": 32,
            "dropout_1": 0.3,
            "dropout_2": 0.2,
            "seq_length": seq_length,
        }

    @classmethod
    def build_model_from_config(cls, config: Dict[str, Any]) -> SimpleRNNClassifier:
        """Instantiate SimpleRNNClassifier from configuration dictionary."""
        return SimpleRNNClassifier(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_units=config["hidden_units"],
            dense_units=config.get("dense_units", 32),
            dropout_1=config.get("dropout_1", 0.3),
            dropout_2=config.get("dropout_2", 0.2),
        )
