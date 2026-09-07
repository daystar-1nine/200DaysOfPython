"""
Sample size determination for targeted margins of error.
"""
import math
from scipy import stats
try:
    from app.validator import validate_confidence_level, validate_margin_of_error
except (ImportError, ModuleNotFoundError):
    from validator import validate_confidence_level, validate_margin_of_error

class SampleSizeCalculator:
    @staticmethod
    def for_mean(
        population_std: float,
        margin_of_error: float,
        confidence: float = 0.95
    ) -> int:
        if population_std <= 0:
            raise ValueError("Population standard deviation must be strictly positive.")
        me = validate_margin_of_error(margin_of_error)
        conf = validate_confidence_level(confidence)
        
        alpha = 1.0 - conf
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        n_exact = (z_crit * population_std / me) ** 2
        return math.ceil(n_exact)

    @staticmethod
    def for_proportion(
        margin_of_error: float,
        confidence: float = 0.95,
        estimated_p: float = 0.5
    ) -> int:
        me = validate_margin_of_error(margin_of_error)
        conf = validate_confidence_level(confidence)
        if not (0.0 < estimated_p < 1.0):
            raise ValueError("Estimated proportion must be between 0 and 1.")
            
        alpha = 1.0 - conf
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        n_exact = (z_crit**2 * estimated_p * (1.0 - estimated_p)) / (me**2)
        return math.ceil(n_exact)
