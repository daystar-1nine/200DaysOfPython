# Day 35: Generators & Large File Processor

A memory-efficient Python large file processor and streaming pipeline built with generators (`yield`).

## Features & Generators

- **`read_lines(filename)` & `read_csv_records(filename)`:** Stream file records lazily with $O(1)$ RAM footprint.
- **`parse_records(records)` & `filter_positive_amounts(records)`:** Clean and transform stream data lazily.
- **`DataPipeline(filename)`:** Chains reader, parser, and filters into a low-memory pipeline.
- **`benchmark_memory_and_speed(count)`:** Demonstrates ~40,000x RAM reduction comparing eager lists vs. lazy generators.

## Execution & Testing

```bash
# Run memory benchmark demo
python Day\ 35/processor/generator_demo.py

# Run streaming pipeline demo
python Day\ 35/processor/pipeline.py

# Run Pytest suite
pytest Day\ 35/tests/test_processor.py
```
