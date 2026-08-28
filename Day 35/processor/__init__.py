"""Day 35 Large File Processor Package."""
import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from reader import read_lines, read_csv_records
from filters import filter_positive_amounts, filter_by_category, parse_records
from pipeline import DataPipeline
from generator_demo import fibonacci_generator, even_numbers_generator, squares_generator

__all__ = [
    "read_lines",
    "read_csv_records",
    "filter_positive_amounts",
    "filter_by_category",
    "parse_records",
    "DataPipeline",
    "fibonacci_generator",
    "even_numbers_generator",
    "squares_generator"
]
