"""
Advanced Customer Analytics Report Engine — CLI Main Entry Point
================================================================
Orchestrates data ingestion, cleaning, statistical profiling,
generation of all 17 publication-grade charts, and executive report authoring.
"""

import sys
import os
import time
from app.config import DATA_PATH, CHARTS_DIR, REPORT_PATH
from app.loader import load_dataset
from app.cleaner import clean_sales_data
from app.visualizations import generate_all_visualizations
from app.report import generate_advanced_eda_report

def run_pipeline():
    """
    Executes the full Day 64 Advanced Statistical EDA Pipeline.
    """
    print("=" * 75)
    print(">>> STARTING DAY 64: ADVANCED SEABORN & STATISTICAL EDA PIPELINE")
    print("=" * 75)
    start_time = time.time()

    # Step 1: Ingestion
    print(f"[1/4] Ingesting dataset from: {DATA_PATH} ...")
    raw_df = load_dataset(DATA_PATH)
    print(f"      Successfully ingested {len(raw_df):,} raw records.")

    # Step 2: Data Cleaning
    print("[2/4] Preprocessing, normalizing types and verifying columns ...")
    clean_df = clean_sales_data(raw_df)
    print(f"      Clean dataset contains {len(clean_df):,} rows across {len(clean_df.columns)} dimensions.")

    # Step 3: Render All 17 Visualizations
    print(f"[3/4] Generating all 17 publication-grade figures in: {CHARTS_DIR} ...")
    charts = generate_all_visualizations(clean_df, CHARTS_DIR)
    print(f"      Successfully rendered {len(charts)} statistical figures at 300 DPI.")

    # Step 4: Executive Report
    print(f"[4/4] Authoring 9-Section Advanced EDA Report: {REPORT_PATH} ...")
    report_text = generate_advanced_eda_report(clean_df, REPORT_PATH)
    print(f"      Report generated successfully ({len(report_text):,} characters).")

    elapsed = time.time() - start_time
    print("=" * 75)
    print(f"[SUCCESS] DAY 64 ADVANCED EDA PIPELINE COMPLETED IN {elapsed:.2f}s")
    print("=" * 75)

if __name__ == "__main__":
    run_pipeline()
