"""
Statistical EDA Visualization Engine — CLI Main Entry Point
============================================================
Orchestrates end-to-end data ingestion, cleaning, statistical profiling,
generation of all 12 publication charts, and executive report authoring.
"""

import sys
import os
import time
from app.config import DATA_PATH, CHARTS_DIR, REPORT_PATH
from app.loader import load_dataset
from app.cleaner import clean_sales_data
from app.charts import generate_all_charts
from app.report import generate_statistical_report

def run_pipeline():
    """
    Executes the full Day 63 statistical visualization workflow.
    """
    print("=" * 70)
    print(">>> STARTING DAY 63: SEABORN STATISTICAL EDA PIPELINE")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Load Data
    print(f"[1/4] Loading dataset from: {DATA_PATH} ...")
    raw_df = load_dataset(DATA_PATH)
    print(f"      Loaded {len(raw_df):,} raw records across {len(raw_df.columns)} columns.")

    # Step 2: Clean and Preprocess
    print("[2/4] Preprocessing and deriving customer segments ...")
    clean_df = clean_sales_data(raw_df)
    print(f"      Cleaned dataset contains {len(clean_df):,} rows. Segment breakdown:")
    for seg, count in clean_df["Customer_Segment"].value_counts().items():
        print(f"        - {seg:<12}: {count:,} orders")

    # Step 3: Generate All 12 Statistical Visualizations
    print(f"[3/4] Generating 12 publication-grade figures in: {CHARTS_DIR} ...")
    charts = generate_all_charts(clean_df, CHARTS_DIR)
    print(f"      Successfully rendered {len(charts)} charts at 300 DPI.")

    # Step 4: Generate Statistical Executive Report
    print(f"[4/4] Authoring 10-Question Statistical Executive Report: {REPORT_PATH} ...")
    report_text = generate_statistical_report(clean_df, REPORT_PATH)
    print(f"      Report written ({len(report_text)} characters).")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[SUCCESS] DAY 63 PIPELINE COMPLETE IN {elapsed:.2f} SECONDS")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
