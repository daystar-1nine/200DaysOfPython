"""
Master Visualization Pipeline Orchestrator
==========================================
Coordinates data handoff from analyzer to specialized plotting modules,
generating all 17 publication-quality charts.
"""

import os
from app.config import CHARTS_DIR
from app.analyzer import (
    get_regional_revenue_stats,
    get_category_profit_stats,
    get_region_counts,
    get_correlation_matrix,
    get_top_customers
)
from app.distributions import (
    plot_revenue_distribution,
    plot_profit_distribution,
    plot_quantity_distribution
)
from app.categorical import (
    plot_revenue_region_boxplot,
    plot_profit_category_boxplot,
    plot_quantity_category_violin,
    plot_regional_revenue,
    plot_category_profit,
    plot_region_counts
)
from app.relationships import (
    plot_region_category_revenue,
    plot_faceted_category_analysis,
    plot_revenue_profit,
    plot_category_relationships
)
from app.time_analysis import (
    plot_monthly_revenue,
    plot_monthly_profit
)
from app.correlation import (
    plot_correlation_heatmap
)
from app.customer_analysis import (
    plot_top_customers
)

def generate_all_visualizations(df, charts_dir: str = None) -> list[str]:
    """
    Executes all 17 visualization routines.
    """
    out_dir = charts_dir if charts_dir is not None else CHARTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 1. Precalculate business data
    regional_df = get_regional_revenue_stats(df)
    category_df = get_category_profit_stats(df)
    counts_df = get_region_counts(df)
    corr_df = get_correlation_matrix(df)
    top_rev_df, top_prof_df, _ = get_top_customers(df, 10)

    # 2. Execution plan
    chart_tasks = [
        ("revenue_distribution.png", lambda p: plot_revenue_distribution(df, p)),
        ("profit_distribution.png", lambda p: plot_profit_distribution(df, p)),
        ("quantity_distribution.png", lambda p: plot_quantity_distribution(df, p)),
        ("revenue_region_boxplot.png", lambda p: plot_revenue_region_boxplot(df, p)),
        ("profit_category_boxplot.png", lambda p: plot_profit_category_boxplot(df, p)),
        ("quantity_category_violin.png", lambda p: plot_quantity_category_violin(df, p)),
        ("regional_revenue.png", lambda p: plot_regional_revenue(regional_df, p)),
        ("category_profit.png", lambda p: plot_category_profit(category_df, p)),
        ("region_counts.png", lambda p: plot_region_counts(counts_df, p)),
        ("region_category_revenue.png", lambda p: plot_region_category_revenue(df, p)),
        ("faceted_category_analysis.png", lambda p: plot_faceted_category_analysis(df, p)),
        ("revenue_profit.png", lambda p: plot_revenue_profit(df, p)),
        ("category_relationships.png", lambda p: plot_category_relationships(df, p)),
        ("monthly_revenue.png", lambda p: plot_monthly_revenue(df, p)),
        ("monthly_profit.png", lambda p: plot_monthly_profit(df, p)),
        ("correlation_heatmap.png", lambda p: plot_correlation_heatmap(corr_df, p)),
        ("top_customers.png", lambda p: plot_top_customers(top_rev_df, top_prof_df, p))
    ]

    generated_paths = []
    for filename, fn in chart_tasks:
        dest_path = os.path.join(out_dir, filename)
        fn(dest_path)
        generated_paths.append(dest_path)
        print(f"Generated visualization: {filename}")

    return generated_paths
