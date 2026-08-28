# ==============================================================================
# Program    : Generator Pipeline Orchestrator (pipeline.py)
# Objective  : Chain generator reader, parser, and filters into a streaming pipeline.
# Concept    : Generator Pipeline Architecture
# Why Used   : Connects low-memory streaming stages together seamlessly.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from reader import read_csv_records
from filters import parse_records, filter_positive_amounts, filter_by_category

class DataPipeline:
    def __init__(self, filename: str):
        self.filename = filename

    def process(self, category: str | None = None):
        """Constructs and returns streaming generator pipeline."""
        # 1. Reader generator
        records_stream = read_csv_records(self.filename)
        # 2. Parser generator
        parsed_stream = parse_records(records_stream)
        # 3. Positive amounts filter
        valid_stream = filter_positive_amounts(parsed_stream)
        # 4. Optional Category filter
        if category:
            valid_stream = filter_by_category(valid_stream, category)
        
        return valid_stream


if __name__ == "__main__":
    data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_large.csv"))
    pipeline = DataPipeline(data_file)
    print("=== STREAMING DATA PIPELINE RESULTS (Food Category) ===")
    for record in pipeline.process(category="Food"):
        print(f"Record #{record['id']}: Rs.{record['amount']:.2f} -> {record['description']}")
