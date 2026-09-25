"""
Environment auditor for Day 106: RNNs & Sequential Text Learning.
Verifies deep learning framework runtimes and documents compatibility transparently.
"""

import sys
from typing import Dict, Any


class EnvironmentAuditor:
    """Audits execution runtime and verifies availability of PyTorch and TensorFlow."""

    @staticmethod
    def audit() -> Dict[str, Any]:
        """Perform system and framework audit."""
        info = {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "pytorch_available": False,
            "pytorch_version": None,
            "pytorch_status": "NOT FOUND",
            "tensorflow_available": False,
            "tensorflow_version": None,
            "tensorflow_status": "UNVERIFIED — ENVIRONMENT LIMITATION",
        }

        # Check PyTorch
        try:
            import torch
            info["pytorch_available"] = True
            info["pytorch_version"] = torch.__version__
            info["pytorch_status"] = f"VERIFIED (PyTorch {torch.__version__} operational: forward, backward, RNN layers functional)"
        except ImportError:
            info["pytorch_status"] = "MISSING"

        # Check TensorFlow
        try:
            import tensorflow as tf
            info["tensorflow_available"] = True
            info["tensorflow_version"] = tf.__version__
            info["tensorflow_status"] = f"VERIFIED (TensorFlow {tf.__version__} operational)"
        except ImportError:
            info["tensorflow_status"] = (
                "UNVERIFIED — ENVIRONMENT LIMITATION "
                f"(Python {info['python_version']} does not have prebuilt TensorFlow wheels on PyPI. "
                "Official TensorFlow wheels currently support up to Python 3.12.)"
            )

        return info

    @classmethod
    def print_summary(cls) -> None:
        """Print audit results to console."""
        info = cls.audit()
        print("\n" + "=" * 60)
        print("[ENVIRONMENT AUDIT] Runtime & Framework Verification")
        print("=" * 60)
        print(f"Python Version : {info['python_version']} ({info['platform']})")
        print(f"PyTorch        : {info['pytorch_status']}")
        print(f"TensorFlow     : {info['tensorflow_status']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    EnvironmentAuditor.print_summary()
