"""
Model definitions: classical TF-IDF baselines and neural text classifiers.
"""

from .baseline import TfidfBaseline
from .embedding_classifier import NeuralTextClassifier
from .architectures import build_model_a, build_model_b, build_model_c

__all__ = ["TfidfBaseline", "NeuralTextClassifier", "build_model_a", "build_model_b", "build_model_c"]
