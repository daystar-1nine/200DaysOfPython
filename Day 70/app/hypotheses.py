"""
Hypothesis specification structures and results.
"""

from dataclasses import dataclass
from typing import Optional

try:
    from app.config import AlternativeType, TestKind
except (ImportError, ModuleNotFoundError):
    from config import AlternativeType, TestKind

@dataclass
class HypothesisSpec:
    test_id: str
    scenario_name: str
    test_kind: TestKind
    parameter_name: str
    null_value: float
    alternative: AlternativeType
    alpha: float = 0.05
    description: str = ""

@dataclass
class TestResult:
    __test__ = False
    spec: HypothesisSpec
    sample_size: int
    point_estimate: float
    standard_error: float
    test_statistic: float
    critical_value: float
    p_value: float
    ci_lower: float
    ci_upper: float
    cohens_d: Optional[float]
    effect_magnitude: Optional[str]
    reject_null: bool
    decision_pvalue: str
    decision_critical: str
    decision_ci: str
    summary: str
