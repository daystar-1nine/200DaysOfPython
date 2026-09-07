"""
Confidence Interval calculation engine for means and proportions.
"""
from typing import Optional
import numpy as np
from scipy import stats
try:
    from app.validator import validate_numeric_array, validate_confidence_level, validate_proportion_counts
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array, validate_confidence_level, validate_proportion_counts

class ConfidenceIntervalEngine:
    @staticmethod
    def z_interval_mean(
        mean: float,
        population_std: float,
        sample_size: int,
        confidence: float = 0.95
    ) -> dict:
        if sample_size <= 0:
            raise ValueError("Sample size must be strictly positive.")
        if population_std < 0:
            raise ValueError("Population standard deviation cannot be negative.")
        conf = validate_confidence_level(confidence)
        
        alpha = 1.0 - conf
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        se = float(population_std / np.sqrt(sample_size))
        me = float(z_crit * se)
        
        return {
            "method": "Z-Interval (Known Sigma)",
            "sample_size": sample_size,
            "sample_mean": round(mean, 4),
            "critical_value": round(z_crit, 4),
            "standard_error": round(se, 4),
            "margin_of_error": round(me, 4),
            "lower_bound": round(mean - me, 4),
            "upper_bound": round(mean + me, 4),
            "confidence_level": conf,
            "interval_width": round(2 * me, 4)
        }

    @staticmethod
    def t_interval_mean(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> dict:
        arr = validate_numeric_array(data)
        n = len(arr)
        if n < 2:
            raise ValueError("At least 2 data points required for t-interval.")
        conf = validate_confidence_level(confidence)
        
        df = n - 1
        mean_val = float(np.mean(arr))
        s = float(np.std(arr, ddof=1))
        se = float(s / np.sqrt(n))
        
        alpha = 1.0 - conf
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        me = float(t_crit * se)
        
        return {
            "method": "T-Interval (Unknown Sigma)",
            "sample_size": n,
            "degrees_of_freedom": df,
            "sample_mean": round(mean_val, 4),
            "sample_std": round(s, 4),
            "critical_value": round(t_crit, 4),
            "standard_error": round(se, 4),
            "margin_of_error": round(me, 4),
            "lower_bound": round(mean_val - me, 4),
            "upper_bound": round(mean_val + me, 4),
            "confidence_level": conf,
            "interval_width": round(2 * me, 4)
        }

    @staticmethod
    def adaptive_mean_interval(
        data: np.ndarray,
        confidence: float = 0.95,
        population_std: Optional[float] = None
    ) -> dict:
        arr = validate_numeric_array(data)
        if population_std is not None:
            return ConfidenceIntervalEngine.z_interval_mean(
                mean=float(np.mean(arr)),
                population_std=population_std,
                sample_size=len(arr),
                confidence=confidence
            )
        return ConfidenceIntervalEngine.t_interval_mean(arr, confidence=confidence)

    @staticmethod
    def proportion_interval(
        successes: int,
        total: int,
        confidence: float = 0.95,
        method: str = "wilson"
    ) -> dict:
        x, n = validate_proportion_counts(successes, total)
        conf = validate_confidence_level(confidence)
        p_hat = float(x / n)
        alpha = 1.0 - conf
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        
        if method == "wald":
            se = float(np.sqrt(p_hat * (1.0 - p_hat) / n))
            me = float(z_crit * se)
            lower = max(0.0, p_hat - me)
            upper = min(1.0, p_hat + me)
        elif method == "wilson":
            denom = 1.0 + (z_crit**2) / n
            center = (p_hat + (z_crit**2) / (2.0 * n)) / denom
            spread = (z_crit * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z_crit**2) / (4.0 * n**2))) / denom
            lower = max(0.0, center - spread)
            upper = min(1.0, center + spread)
            se = float(np.sqrt(p_hat * (1.0 - p_hat) / n))
            me = float(spread)
        else:
            raise ValueError(f"Unknown method '{method}'. Supported: 'wald', 'wilson'.")
            
        return {
            "method": f"Proportion ({method.title()})",
            "successes": x,
            "sample_size": n,
            "sample_proportion": round(p_hat, 4),
            "critical_value": round(z_crit, 4),
            "standard_error": round(se, 4),
            "margin_of_error": round(me, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "confidence_level": conf,
            "interval_width": round(upper - lower, 4)
        }
