"""
Unit tests for hypothesis specs and enumerations.
"""

try:
    from app.config import AlternativeType, TestKind
    from app.hypotheses import HypothesisSpec, TestResult
except (ImportError, ModuleNotFoundError):
    from config import AlternativeType, TestKind
    from hypotheses import HypothesisSpec, TestResult

def test_alternative_type_enum():
    assert AlternativeType.TWO_SIDED.value == "two-sided"
    assert AlternativeType.GREATER.value == "greater"
    assert AlternativeType.LESS.value == "less"

def test_test_kind_enum():
    assert TestKind.ONE_SAMPLE_T.value == "one_sample_t"
    assert TestKind.ONE_SAMPLE_Z.value == "one_sample_z"
    assert TestKind.PROPORTION_Z.value == "proportion_z"

def test_hypothesis_spec_creation():
    spec = HypothesisSpec(
        test_id="TEST_01",
        scenario_name="Test Scenario",
        test_kind=TestKind.ONE_SAMPLE_T,
        parameter_name="mean",
        null_value=100.0,
        alternative=AlternativeType.GREATER,
        alpha=0.05
    )
    assert spec.test_id == "TEST_01"
    assert spec.null_value == 100.0
    assert spec.alpha == 0.05
