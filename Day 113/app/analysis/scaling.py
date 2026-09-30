"""
Empirical Scaling Law Modeling (Kaplan et al. / Chinchilla).
Models power-law decay of cross-entropy loss with respect to parameters and tokens:
    L(N) = E + A * N^(-alpha)
"""
import math
from typing import List, Dict, Tuple, Any
import numpy as np


def power_law_loss(scale: float, a: float, alpha: float, irreducible_loss: float = 1.6) -> float:
    """
    Computes theoretical loss under power-law scaling:
      L = E + A * scale^(-alpha)
    """
    if scale <= 0:
        raise ValueError("Scale must be strictly positive.")
    return float(irreducible_loss + a * (scale ** (-alpha)))


def fit_power_law(
    scales: List[float],
    losses: List[float],
    irreducible_loss: float = 1.6
) -> Dict[str, float]:
    """
    Fits power-law parameters A and alpha using log-linear regression:
      log(L - E) = log(A) - alpha * log(N)
    """
    if len(scales) != len(losses) or len(scales) < 2:
        raise ValueError("Must provide at least two (scale, loss) observations to fit power law.")

    adjusted_losses = [max(l - irreducible_loss, 1e-4) for l in losses]
    log_scales = np.log(np.array(scales, dtype=np.float64))
    log_losses = np.log(np.array(adjusted_losses, dtype=np.float64))

    # Linear fit: y = m*x + c, where m = -alpha, c = log(A)
    poly = np.polyfit(log_scales, log_losses, 1)
    neg_alpha, log_a = poly[0], poly[1]

    alpha = float(-neg_alpha)
    a = float(np.exp(log_a))

    return {
        "A": round(a, 4),
        "alpha": round(alpha, 4),
        "irreducible_loss": round(irreducible_loss, 4)
    }


def predict_scaled_loss(
    target_scale: float,
    a: float,
    alpha: float,
    irreducible_loss: float = 1.6
) -> float:
    """Predicts expected validation loss for an unobserved scale."""
    return round(power_law_loss(target_scale, a, alpha, irreducible_loss), 4)


def analyze_scaling_efficiency(
    model_sizes: List[int],
    val_losses: List[float]
) -> List[Dict[str, Any]]:
    """
    Analyzes marginal loss reduction per doubling of parameters:
      Delta_L / Delta_log2(N)
    """
    results = []
    for i in range(len(model_sizes)):
        entry = {
            "parameters": model_sizes[i],
            "loss": val_losses[i],
            "marginal_improvement": None
        }
        if i > 0:
            delta_loss = val_losses[i - 1] - val_losses[i]
            param_ratio = model_sizes[i] / model_sizes[i - 1]
            doublings = math.log2(param_ratio) if param_ratio > 1.0 else 1.0
            entry["marginal_improvement"] = round(delta_loss / doublings, 4)
        results.append(entry)
    return results
