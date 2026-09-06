"""
Theoretical probability distribution analyzer.
"""
import sys
from pathlib import Path
day67_dir = Path(__file__).resolve().parent.parent
if str(day67_dir) not in sys.path:
    sys.path.insert(0, str(day67_dir))

from typing import Any
import pandas as pd
from distributions.base import BaseDistribution
from distributions.bernoulli import BernoulliDistribution
from distributions.binomial import BinomialDistribution
from distributions.uniform import UniformDistribution
from distributions.normal import NormalDistribution
from distributions.poisson import PoissonDistribution
from app.validator import validate_distribution_params

def create_distribution(name: str, params: dict[str, Any]) -> BaseDistribution:
    """Factory function instantiating validated distribution objects."""
    validate_distribution_params(name, params)
    clean_name = name.strip().lower()
    
    if clean_name == "bernoulli":
        return BernoulliDistribution(params["p"])
    elif clean_name == "binomial":
        return BinomialDistribution(params["n"], params["p"])
    elif clean_name == "uniform":
        return UniformDistribution(params["a"], params["b"])
    elif clean_name == "normal":
        return NormalDistribution(params["mu"], params["sigma"])
    elif clean_name == "poisson":
        lam = params.get("lambda", params.get("mu"))
        return PoissonDistribution(lam)
    else:
        raise ValueError(f"Unknown distribution '{name}'.")

def evaluate_distribution_probabilities(dist: BaseDistribution) -> pd.DataFrame:
    """Compute standard probability queries (point/density, CDF, tail probabilities, quantiles)."""
    records = []
    
    if dist.name == "Bernoulli":
        for k in [0, 1]:
            records.append({
                "Distribution": dist.name,
                "Query": f"P(X = {k})",
                "Value": round(float(dist.pmf_or_pdf(k)), 6)
            })
            
    elif dist.name == "Binomial":
        n = dist.parameters["n"]
        test_points = [int(n * 0.25), int(n * 0.50), int(n * 0.75)]
        for k in test_points:
            records.append({
                "Distribution": dist.name,
                "Query": f"P(X = {k})",
                "Value": round(float(dist.pmf_or_pdf(k)), 6)
            })
            records.append({
                "Distribution": dist.name,
                "Query": f"P(X <= {k})",
                "Value": round(float(dist.cdf(k)), 6)
            })
            records.append({
                "Distribution": dist.name,
                "Query": f"P(X >= {k})",
                "Value": round(float(1.0 - dist.cdf(k - 1)), 6)
            })
            
    elif dist.name == "Uniform":
        a, b = dist.parameters["a"], dist.parameters["b"]
        mid = (a + b) / 2.0
        records.append({"Distribution": dist.name, "Query": f"PDF at Midpoint ({mid})", "Value": round(float(dist.pmf_or_pdf(mid)), 6)})
        records.append({"Distribution": dist.name, "Query": f"P(X <= {mid})", "Value": round(float(dist.cdf(mid)), 6)})
        records.append({"Distribution": dist.name, "Query": f"P({a + (b-a)*0.25:.1f} <= X <= {a + (b-a)*0.75:.1f})", "Value": round(float(dist.cdf(a + (b-a)*0.75) - dist.cdf(a + (b-a)*0.25)), 6)})
        
    elif dist.name == "Normal":
        mu, sigma = dist.parameters["mu"], dist.parameters["sigma"]
        records.append({"Distribution": dist.name, "Query": f"P(X <= {mu - sigma:.1f}) [Below 1s]", "Value": round(float(dist.cdf(mu - sigma)), 6)})
        records.append({"Distribution": dist.name, "Query": f"P({mu - sigma:.1f} <= X <= {mu + sigma:.1f}) [Within 1s]", "Value": round(float(dist.cdf(mu + sigma) - dist.cdf(mu - sigma)), 6)})
        records.append({"Distribution": dist.name, "Query": f"P({mu - 2*sigma:.1f} <= X <= {mu + 2*sigma:.1f}) [Within 2s]", "Value": round(float(dist.cdf(mu + 2*sigma) - dist.cdf(mu - 2*sigma)), 6)})
        records.append({"Distribution": dist.name, "Query": "95th Percentile (PPF 0.95)", "Value": round(float(dist.ppf(0.95)), 4)})
        records.append({"Distribution": dist.name, "Query": "99th Percentile (PPF 0.99)", "Value": round(float(dist.ppf(0.99)), 4)})
        
    elif dist.name == "Poisson":
        lam = dist.parameters["lambda"]
        k_val = int(lam)
        records.append({"Distribution": dist.name, "Query": f"P(X = {k_val}) [Peak]", "Value": round(float(dist.pmf_or_pdf(k_val)), 6)})
        records.append({"Distribution": dist.name, "Query": f"P(X <= {k_val})", "Value": round(float(dist.cdf(k_val)), 6)})
        records.append({"Distribution": dist.name, "Query": f"P(X > {k_val})", "Value": round(float(1.0 - dist.cdf(k_val)), 6)})
        
    return pd.DataFrame(records)
