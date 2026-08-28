# ==============================================================================
# Program    : Report Service & Exporter Protocol (report_service.py)
# Objective  : Export log reports using LogExporter Protocol (Day 39) & custom exceptions (Day 37).
# Concept    : Structural Subtyping Exporter & Report Exporter Service
# Why Used   : Decouples report formatting from specific file export targets.
# ==============================================================================

import json
import os
import sys
from typing import Protocol, runtime_checkable

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.exceptions import ReportGenerationError
from loganalyze.models.log_entry import LogLevel
from loganalyze.models.report import LogReport

# What is used : Protocol Interface (Day 39 requirement)
# Why it is used: Defines duck-typing structural contract for report exporters
@runtime_checkable
class LogExporter(Protocol):
    """Protocol interface defining required method for log exporters."""
    def export(self, report: LogReport, output_path: str) -> None:
        ...

class JSONReportExporter:
    """JSON Report Exporter satisfying LogExporter Protocol."""
    def export(self, report: LogReport, output_path: str) -> None:
        try:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            data = {
                "total_lines": report.total_lines,
                "counts": {
                    LogLevel.INFO.value: report.counts.get(LogLevel.INFO, 0),
                    LogLevel.WARNING.value: report.counts.get(LogLevel.WARNING, 0),
                    LogLevel.ERROR.value: report.counts.get(LogLevel.ERROR, 0)
                },
                "errors": [
                    {
                        "timestamp": e.timestamp,
                        "level": e.level.value,
                        "message": e.message
                    }
                    for e in report.error_entries
                ]
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise ReportGenerationError(f"Failed exporting report to '{output_path}': {e}") from e

class ReportService:
    """Service handling report exports using LogExporter strategy."""
    def __init__(self, exporter: LogExporter | None = None):
        self.exporter = exporter or JSONReportExporter()

    def export_report(self, report: LogReport, output_path: str) -> None:
        self.exporter.export(report, output_path)
