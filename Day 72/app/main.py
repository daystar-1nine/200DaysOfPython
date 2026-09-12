"""
Day 72 — Business Relationship & Correlation Analyzer CLI Entry Point
Orchestrates data ingestion, validation, matrix generation, statistical analysis, charts, and reporting.
"""
import os
import sys

try:
    from app.config import AppConfig
    from app.loader import load_dataset
    from app.cleaner import clean_dataset
    from app.validator import validate_numeric_data
    from app.covariance import compute_covariance_matrix
    from app.correlation_matrix import compute_correlation_matrices, compute_all_pairwise_results
    from app.analyzer import RelationshipAnalyzer
    from app.insights import generate_business_insights
    from app.visualizations import generate_all_charts
    from app.report import export_csv_results, format_ascii_report
except ImportError:
    from config import AppConfig
    from loader import load_dataset
    from cleaner import clean_dataset
    from validator import validate_numeric_data
    from covariance import compute_covariance_matrix
    from correlation_matrix import compute_correlation_matrices, compute_all_pairwise_results
    from analyzer import RelationshipAnalyzer
    from insights import generate_business_insights
    from visualizations import generate_all_charts
    from report import export_csv_results, format_ascii_report

def run_pipeline(csv_path: str = None) -> int:
    config = AppConfig()
    if csv_path:
        config.DATA_PATH_CSV = csv_path
        
    print("=" * 70)
    print("  BUSINESS RELATIONSHIP & CORRELATION ANALYZER PIPELINE  ")
    print("=" * 70)
    print(f"Loading data from: {config.DATA_PATH_CSV}")
    
    # 1. Load and clean
    raw_df = load_dataset(config.DATA_PATH_CSV)
    cleaned_df = clean_dataset(raw_df)
    
    # 2. Validate
    val_info = validate_numeric_data(cleaned_df)
    print(f"Validated numerical dataset: {val_info['n_rows']} rows, {val_info['n_cols']} features.")
    
    # 3. Covariance and Correlation Matrices
    print("Computing covariance and correlation matrices...")
    cov_mat = compute_covariance_matrix(cleaned_df)
    pearson_mat, spearman_mat, diff_mat = compute_correlation_matrices(cleaned_df)
    
    # 4. Pairwise Analysis
    print("Evaluating pairwise parametric and non-parametric statistics...")
    pairwise_df = compute_all_pairwise_results(cleaned_df, config=config)
    
    # 5. High-level analysis
    analyzer = RelationshipAnalyzer(pairwise_df=pairwise_df, raw_df=cleaned_df, config=config)
    top_pos = analyzer.get_top_positive(config.TOP_N)
    top_neg = analyzer.get_top_negative(config.TOP_N)
    mc_df = analyzer.get_multicollinear_pairs()
    div_df = analyzer.get_divergent_pairs()
    q_and_a = analyzer.answer_business_questions()
    
    # 6. Automated Insights
    insights = generate_business_insights(top_pos, top_neg, mc_df)
    
    # 7. Render Visualizations
    print("Rendering publication-grade visualizations...")
    generate_all_charts(cleaned_df, pearson_mat, pairwise_df, config.OUTPUT_DIR)
    
    # 8. Export CSV and Text Reports
    print("Exporting matrices and analytical reports...")
    export_csv_results(cov_mat, pearson_mat, pairwise_df, config.OUTPUT_DIR)
    
    report_text = format_ascii_report(
        n_rows=val_info["n_rows"],
        n_cols=val_info["n_cols"],
        top_pos=top_pos,
        top_neg=top_neg,
        mc_df=mc_df,
        divergent_df=div_df,
        insights=insights,
        q_and_a=q_and_a
    )
    
    with open(config.REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"Report successfully saved to: {config.REPORT_TXT}")
    print("\n" + report_text)
    print("\nPipeline execution completed successfully!")
    return 0

if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_pipeline(path_arg))
