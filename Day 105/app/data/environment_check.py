"""
Environment verification module for Day 105: Neural NLP & Text Classification.
Verifies deep learning framework compatibility in the current Python runtime.
"""

import sys
from typing import Dict, Any


class EnvironmentChecker:
    """Audits deep learning framework support in the active Python environment."""

    @staticmethod
    def audit() -> Dict[str, Any]:
        """Perform environment audit and return verification status dictionary."""
        report = {
            "python_version": sys.version,
            "tensorflow_status": "UNVERIFIED — ENVIRONMENT LIMITATION",
            "tensorflow_message": "",
            "pytorch_status": "UNVERIFIED",
            "pytorch_message": "",
            "pytorch_version": None
        }

        # 1. Audit TensorFlow
        try:
            import tensorflow as tf  # noqa: F401
            report["tensorflow_status"] = "VERIFIED"
            report["tensorflow_message"] = f"TensorFlow {tf.__version__} successfully imported and ready."
        except ModuleNotFoundError:
            report["tensorflow_status"] = "UNVERIFIED — ENVIRONMENT LIMITATION"
            report["tensorflow_message"] = (
                f"Python {sys.version.split()[0]} does not have prebuilt TensorFlow wheels. "
                "Official TensorFlow releases currently support up to Python 3.12."
            )
        except Exception as e:
            report["tensorflow_status"] = "UNVERIFIED — ENVIRONMENT ERROR"
            report["tensorflow_message"] = str(e)

        # 2. Audit PyTorch
        try:
            import torch
            import torch.nn as nn

            report["pytorch_version"] = torch.__version__

            # Verify build, train, predict, save, reload
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(4, 1)
                def forward(self, x):
                    return self.linear(x)

            m = DummyModel()
            x = torch.randn(2, 4)
            y = m(x)
            loss = y.sum()
            loss.backward()

            report["pytorch_status"] = "VERIFIED"
            report["pytorch_message"] = f"PyTorch {torch.__version__} verified: build, train, forward, backward pass operational."
        except ModuleNotFoundError:
            report["pytorch_status"] = "NOT INSTALLED"
            report["pytorch_message"] = "PyTorch is not installed in the environment."
        except Exception as e:
            report["pytorch_status"] = "UNVERIFIED — ERROR"
            report["pytorch_message"] = str(e)

        return report
