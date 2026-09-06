"""Statistics computation modules."""
import os
import sys
import importlib.util

_day_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _day_dir not in sys.path:
    sys.path.insert(0, _day_dir)

# Re-export stdlib attributes so external packages (such as Seaborn) that do
# `from statistics import NormalDist` succeed even if this package shadows stdlib statistics
_stdlib_file = os.path.join(os.path.dirname(os.__file__), "statistics.py")
if os.path.exists(_stdlib_file):
    _spec = importlib.util.spec_from_file_location("_stdlib_statistics", _stdlib_file)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    for _attr in dir(_mod):
        if not _attr.startswith("_") and _attr not in globals():
            globals()[_attr] = getattr(_mod, _attr)

try:
    from .central_tendency import calculate_mean, calculate_median, calculate_mode
    from .dispersion import calculate_range, calculate_variance, calculate_std, calculate_iqr
    from .percentiles import calculate_percentile, calculate_percentiles
    from .outliers import detect_iqr_outliers
    from .zscore import calculate_zscores, detect_zscore_outliers
    from .distribution import calculate_skewness, calculate_kurtosis, classify_distribution
except ImportError:
    from app.statistics.central_tendency import calculate_mean, calculate_median, calculate_mode
    from app.statistics.dispersion import calculate_range, calculate_variance, calculate_std, calculate_iqr
    from app.statistics.percentiles import calculate_percentile, calculate_percentiles
    from app.statistics.outliers import detect_iqr_outliers
    from app.statistics.zscore import calculate_zscores, detect_zscore_outliers
    from app.statistics.distribution import calculate_skewness, calculate_kurtosis, classify_distribution

