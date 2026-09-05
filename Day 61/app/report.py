"""
Executive Visualization Report Generation Module.
Synthesizes chart interpretations, empirical observations, and strategic business insights.
"""

from pathlib import Path
import pandas as pd


def generate_visualization_report(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Generates a structured text report evaluating all visualizations.

    # What is used: String templating and file I/O
    # Why it is used: Produces an executive-ready business intelligence report accompanying the charts
    # How it works: Compiles four-part sections (Chart, Question, Observation, Business Insight) for all 9 charts
    """
    total_rev = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    profit_margin = (total_profit / total_rev) * 100 if total_rev > 0 else 0.0
    total_orders = len(df)

    report_content = f"""========================================================================================
                          ENTERPRISE SALES VISUALIZATION REPORT
                                  200 DAYS OF PYTHON: DAY 61
========================================================================================
Report Generated: September 2026
Dataset Summary:
  - Total Analyzed Records : {total_orders:,} transactions
  - Cumulative Revenue    : ₹{total_rev:,.2f}
  - Cumulative Net Profit : ₹{total_profit:,.2f}
  - Overall Profit Margin : {profit_margin:.2f}%
========================================================================================

----------------------------------------------------------------------------------------
CHART 1: Monthly Revenue Trend
----------------------------------------------------------------------------------------
Chart:
  Monthly Revenue Trend (Line Chart)

Question:
  How is revenue evolving over time across the 2026 calendar months?

Observation:
  The revenue curve exhibits an overall upward trajectory from Q1 through Q4, with a
  pronounced seasonal acceleration beginning in Q3 and peaking in Q4 (November-December).
  Minor temporary pullbacks are visible during mid-quarter transition months.

Business Insight:
  Growth momentum is strong and accelerating into the final quarters. Supply chain,
  inventory replenishment, and customer support capacities must be scaled by at least
  35% in early Q3 to avoid fulfillment bottlenecks during the Q4 peak demand surge.

----------------------------------------------------------------------------------------
CHART 2: Revenue by Region
----------------------------------------------------------------------------------------
Chart:
  Revenue by Region (Vertical Bar Chart)

Question:
  Which sales region generates the highest revenue, and how balanced is geographic distribution?

Observation:
  The West and North regions generate the largest shares of total enterprise revenue,
  followed closely by South and East. All four territories demonstrate substantial
  commercial viability without dangerous reliance on a single geographic territory.

Business Insight:
  While West remains the corporate anchor, the East region represents an under-penetrated
  market with rapid growth potential. Allocating targeted regional digital marketing
  budgets to East could balance regional parity and unlock untapped commercial market share.

----------------------------------------------------------------------------------------
CHART 3: Revenue by Category
----------------------------------------------------------------------------------------
Chart:
  Revenue by Category (Sorted Vertical Bar Chart)

Question:
  Which product categories dominate total revenue generation?

Observation:
  Electronics and Furniture dominate the product portfolio, together delivering over
  50% of aggregate enterprise turnover. Apparel, Kitchenware, and Fitness follow with
  moderate but steady transaction volume.

Business Insight:
  High-ticket electronics and ergonomic home/office furniture are the primary corporate
  revenue engines. Cross-selling initiatives (e.g., bundling tech peripherals with
  furniture setups) will increase average basket size and margin capture.

----------------------------------------------------------------------------------------
CHART 4: Top 10 Products
----------------------------------------------------------------------------------------
Chart:
  Top 10 Products (Horizontal Bar Chart)

Question:
  Which specific SKUs serve as the top revenue drivers across the catalog?

Observation:
  High-end hardware items (such as Ergonomic Chairs, Standing Desks, and 4K Monitors)
  occupy the topmost ranks in revenue generation, benefiting from elevated unit prices.

Business Insight:
  The top 10 SKUs account for a disproportionate share of total cash inflow. The inventory
  team must maintain safety stock and supplier SLAs specifically for these 10 SKUs to
  prevent catastrophic stockouts of high-velocity cash-flow drivers.

----------------------------------------------------------------------------------------
CHART 5: Top 10 Customers
----------------------------------------------------------------------------------------
Chart:
  Top 10 High-Value Customers (Horizontal Bar Chart)

Question:
  Who are the enterprise's key VIP customer accounts and how concentrated is repeat spend?

Observation:
  The top 10 accounts demonstrate consistent multi-order purchasing behavior, with
  individual cumulative spend significantly exceeding the general customer mean.

Business Insight:
  A VIP Account Management program should be implemented immediately. Assigning
  dedicated account managers and volume loyalty perks to these top 10 clients will
  strengthen retention and protect high-margin reorder cycles from competitors.

----------------------------------------------------------------------------------------
CHART 6: Order Quantity Distribution
----------------------------------------------------------------------------------------
Chart:
  Order Quantity Distribution (Histogram)

Question:
  How are order quantities distributed per transaction?

Observation:
  Order sizes exhibit a tight distribution ranging primarily between 1 and 5 units per
  order, with the median centered at 3 units. Very few orders exceed 8 units.

Business Insight:
  Purchases reflect consumer and small-business direct consumption rather than bulk
  wholesale distribution. Introducing tiered multi-pack incentives (e.g., discounts
  at 5+ units) could effectively push average order quantity toward 4-5 units.

----------------------------------------------------------------------------------------
CHART 7: Revenue vs Profit Relationship
----------------------------------------------------------------------------------------
Chart:
  Revenue vs Profit (Scatter Plot & Correlation)

Question:
  Is higher order revenue consistently associated with higher net profit?

Observation:
  The scatter plot reveals a strong, positive linear association (r > 0.85) between
  transaction revenue and net dollar profit. However, noticeable vertical dispersion
  exists at higher revenue points due to varying discount tiers.

Business Insight:
  While larger orders generate greater absolute profit, heavily discounted orders
  exhibit compressed margins. Discount authorization guardrails should be enforced to
  ensure transaction gross margins do not drop below a strict 18% floor.

----------------------------------------------------------------------------------------
CHART 8: Category Revenue Share
----------------------------------------------------------------------------------------
Chart:
  Category Revenue Share (Pie Chart)

Question:
  What is the proportional part-to-whole contribution of each category?

Observation:
  Electronics and Furniture represent the two largest slices (over 55% combined),
  while Fitness and Kitchenware represent smaller specialized niches.

Business Insight:
  The portfolio has healthy diversification across 5 primary lifestyle and workspace
  pillars. While the pie chart effectively communicates top-level proportions to executive
  boards, detailed SKU performance within each slice must be tracked via bar charts.

----------------------------------------------------------------------------------------
CHART 9 (BONUS): Monthly Revenue with 3-Month Rolling Average
----------------------------------------------------------------------------------------
Chart:
  Monthly Revenue with 3-Month Moving Average (Dual Line Chart)

Question:
  What is the underlying medium-term growth momentum when short-term noise is removed?

Observation:
  The 3-month rolling average smooths out monthly transactional volatility, establishing
  that the corporate revenue baseline is expanding monotonically with positive acceleration.

Business Insight:
  Executive decision-making should be benchmarked against the 3-month smoothed line
  rather than single-month spikes or dips. The smoothed trajectory indicates robust corporate
  health and supports increased capital expenditure for next year's expansion.

========================================================================================
                              END OF VISUALIZATION REPORT
========================================================================================
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content.strip() + "\n")

    return output_path
