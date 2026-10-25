"""
Latency benchmarking utility for measuring retrieval, reranking, and generation runtimes.
"""
from typing import List, Dict, Any, Callable
import time
import numpy as np


class LatencyTracker:
    """
    Collects execution runtimes and computes distribution percentiles (P50, P90, P95, mean, min, max).
    """
    def __init__(self):
        self.measurements: List[float] = []

    def time_call(self, func: Callable, *args, **kwargs) -> Any:
        """Times execution of func in milliseconds."""
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.measurements.append(elapsed_ms)
        return result

    def add_measurement(self, ms: float) -> None:
        """Directly records a measured latency value."""
        self.measurements.append(float(ms))

    def summarize(self) -> Dict[str, float]:
        """Calculates statistical summary percentiles."""
        if not self.measurements:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "median_p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0
            }

        arr = np.array(self.measurements, dtype=np.float64)
        return {
            "count": int(len(arr)),
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "median_p50_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr))
        }
