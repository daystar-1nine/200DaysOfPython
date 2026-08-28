# ==============================================================================
# Program    : LogAnalyze CLI Subcommands Handler (commands.py)
# Objective  : Handle argparse CLI subcommands: summary, errors, search, date, export.
# Concept    : CLI Subcommand Parsing
# Why Used   : Parses command line flags and delegates execution to LogAnalyzer service.
# ==============================================================================

import argparse
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from loganalyze.config import DEFAULT_REPORT_DIR
from loganalyze.exceptions import InvalidLogError
from loganalyze.services.analyzer import LogAnalyzer
from loganalyze.services.report_service import ReportService

def build_parser() -> argparse.ArgumentParser:
    """Builds and returns argparse CLI parser with subcommands."""
    parser = argparse.ArgumentParser(prog="loganalyze", description="Log Analytics CLI Application")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # 1. Summary subcommand
    summary_parser = subparsers.add_parser("summary", help="Generate log summary report")
    summary_parser.add_argument("logfile", help="Path to input log file")

    # 2. Errors subcommand
    errors_parser = subparsers.add_parser("errors", help="List ERROR log entries")
    errors_parser.add_argument("logfile", help="Path to input log file")

    # 3. Search subcommand
    search_parser = subparsers.add_parser("search", help="Search log entries by keyword")
    search_parser.add_argument("logfile", help="Path to input log file")
    search_parser.add_argument("keyword", help="Keyword term to search")

    # 4. Date subcommand
    date_parser = subparsers.add_parser("date", help="Filter log entries by date (YYYY-MM-DD)")
    date_parser.add_argument("logfile", help="Path to input log file")
    date_parser.add_argument("date_str", help="Date in YYYY-MM-DD format")

    # 5. Export subcommand
    export_parser = subparsers.add_parser("export", help="Export log report to JSON file")
    export_parser.add_argument("logfile", help="Path to input log file")
    export_parser.add_argument("output", nargs="?", default="report.json", help="Output JSON filename or path")

    return parser

def parse_and_execute(args: list[str] | None = None) -> None:
    """Parses command line arguments and executes designated subcommand logic."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return

    analyzer = LogAnalyzer()

    if parsed_args.command == "summary":
        report = analyzer.analyze_summary(parsed_args.logfile)
        print(report)

    elif parsed_args.command == "errors":
        errors = analyzer.get_errors(parsed_args.logfile)
        print(f"Found {len(errors)} Error Entries:\n")
        for err in errors:
            print(f"{err.timestamp} | {err.level.value} | {err.message}")

    elif parsed_args.command == "search":
        matches = analyzer.search_logs(parsed_args.logfile, parsed_args.keyword)
        print(f"Found {len(matches)} Matches for '{parsed_args.keyword}':\n")
        for entry in matches:
            print(f"{entry.timestamp} | {entry.level.value} | {entry.message}")

    elif parsed_args.command == "date":
        matches = analyzer.filter_by_date(parsed_args.logfile, parsed_args.date_str)
        print(f"Found {len(matches)} Entries for Date '{parsed_args.date_str}':\n")
        for entry in matches:
            print(f"{entry.timestamp} | {entry.level.value} | {entry.message}")

    elif parsed_args.command == "export":
        report = analyzer.analyze_summary(parsed_args.logfile)
        out_path = parsed_args.output
        if not os.path.isabs(out_path):
            out_path = os.path.join(DEFAULT_REPORT_DIR, out_path)

        report_service = ReportService()
        report_service.export_report(report, out_path)
        print(f"✅ Report successfully exported to: {out_path}")
