"""Application configuration module."""
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW: Path = BASE_DIR / "data" / "raw" / "sms_spam.csv"
    DATA_PROCESSED: Path = BASE_DIR / "data" / "processed" / "cleaned_sms.csv"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    CHARTS_DIR: Path = BASE_DIR / "output" / "charts"
    METRICS_PATH: Path = BASE_DIR / "output" / "metrics.csv"
    PREDICTIONS_PATH: Path = BASE_DIR / "output" / "predictions.csv"
    ERRORS_PATH: Path = BASE_DIR / "output" / "errors.csv"
    FEATURE_ANALYSIS_PATH: Path = BASE_DIR / "output" / "feature_analysis.csv"
    REPORT_PATH: Path = BASE_DIR / "output" / "report.md"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.25
    CV_FOLDS: int = 5

config = Config()
