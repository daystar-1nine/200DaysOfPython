"""
Parameter validation engine for probability distributions.
"""
from typing import Any

def validate_distribution_params(dist_name: str, params: dict[str, Any]) -> None:
    """
    Validate parameter values for supported probability distributions.
    Raises ValueError with descriptive messages if constraints are violated.
    """
    name_clean = dist_name.strip().lower()
    
    if name_clean == "bernoulli":
        if "p" not in params:
            raise ValueError("Bernoulli requires parameter 'p'.")
        p = float(params["p"])
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Bernoulli parameter 'p' must be between 0.0 and 1.0, got {p}.")
            
    elif name_clean == "binomial":
        if "n" not in params or "p" not in params:
            raise ValueError("Binomial requires parameters 'n' and 'p'.")
        n = params["n"]
        p = float(params["p"])
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"Binomial trials 'n' must be a non-negative integer, got {n}.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Binomial probability 'p' must be in [0, 1], got {p}.")
            
    elif name_clean == "uniform":
        if "a" not in params or "b" not in params:
            raise ValueError("Uniform requires parameters 'a' (min) and 'b' (max).")
        a = float(params["a"])
        b = float(params["b"])
        if b <= a:
            raise ValueError(f"Uniform upper bound 'b' ({b}) must be strictly greater than lower bound 'a' ({a}).")
            
    elif name_clean == "normal":
        if "mu" not in params or "sigma" not in params:
            raise ValueError("Normal requires parameters 'mu' and 'sigma'.")
        sigma = float(params["sigma"])
        if sigma <= 0.0:
            raise ValueError(f"Normal standard deviation 'sigma' must be strictly positive (>0), got {sigma}.")
            
    elif name_clean == "poisson":
        if "lambda" not in params and "mu" not in params:
            raise ValueError("Poisson requires parameter 'lambda'.")
        lam = float(params.get("lambda", params.get("mu")))
        if lam < 0.0:
            raise ValueError(f"Poisson rate 'lambda' must be non-negative (>=0), got {lam}.")
            
    else:
        raise ValueError(f"Unsupported distribution: '{dist_name}'. Choose from bernoulli, binomial, uniform, normal, poisson.")
