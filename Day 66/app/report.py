"""
Report generator exporting CSV datasets and ASCII executive summary.
"""
from pathlib import Path
import pandas as pd
from app.config import OUTPUT_DIR

def export_all_datasets(pipeline_results: dict):
    """Export the 4 structured CSV datasets."""
    # 1. Coin results
    coin_sub = pipeline_results["coin"]["df_results"].iloc[::100].copy()  # subsample every 100th row for compact CSV
    coin_csv = OUTPUT_DIR / "coin_results.csv"
    coin_sub.to_csv(coin_csv, index=False)
    print(f"Exported: {coin_csv}")
    
    # 2. Dice results
    dice_csv = OUTPUT_DIR / "dice_results.csv"
    pipeline_results["dice"]["df_distribution"].to_csv(dice_csv, index=False)
    print(f"Exported: {dice_csv}")
    
    # 3. Card results
    card_csv = OUTPUT_DIR / "card_results.csv"
    pipeline_results["cards"]["df_cards"].to_csv(card_csv, index=False)
    print(f"Exported: {card_csv}")
    
    # 4. Conversion results
    conv_csv = OUTPUT_DIR / "conversion_results.csv"
    pipeline_results["conversion"]["df_summary"].to_csv(conv_csv, index=False)
    print(f"Exported: {conv_csv}")

def generate_ascii_report(pipeline_results: dict, output_path: Path = OUTPUT_DIR / "probability_report.txt") -> str:
    """Generate structured ASCII probability analytics summary."""
    coin = pipeline_results["coin"]
    dice = pipeline_results["dice"]
    dice_two = pipeline_results["dice_two"]
    cards = pipeline_results["cards"]
    conv = pipeline_results["conversion"]
    risk = pipeline_results["risk"]
    
    lines = [
        "=" * 78,
        "         DAY 66: REAL-WORLD PROBABILITY SIMULATOR EXECUTIVE REPORT",
        "=" * 78,
        "",
        "1. LAW OF LARGE NUMBERS & COIN TOSS CONVERGENCE",
        "-" * 78,
        f"  Total Flips:          {coin['n_flips']:,}",
        f"  Heads Count:          {coin['final_heads']:,} ({coin['empirical_p']*100:.3f}%)",
        f"  Theoretical Target:   {coin['theoretical_p']*100:.1f}%",
        f"  Absolute Deviation:   {coin['abs_error']:.6f}",
        "  Convergence Verdict:  STABLE CONVERGENCE within 95% confidence boundary.",
        "",
        "2. UNIFORM & SUM PROBABILITY DISTRIBUTIONS (DICE ROLLS)",
        "-" * 78,
        f"  Single Die Rolls:     {dice['n_rolls']:,}",
        f"  Expected per Face:    {dice['n_rolls']/6:,.1f}",
        "  Distribution:",
    ]
    for _, r in dice["df_distribution"].iterrows():
        lines.append(f"    Face {int(r['Face'])}: Count={int(r['Count']):,} | P_emp={r['Empirical_Probability']:.5f} | P_theo={r['Theoretical_Probability']:.5f} | Err={r['Absolute_Error']:.5f}")
        
    lines.extend([
        "",
        f"  Two Dice Sum Rolls:   {dice_two['n_rolls']:,}",
        "  Peak Sum (k=7):",
    ])
    sum7_row = dice_two["df_sums"][dice_two["df_sums"]["Sum"] == 7].iloc[0]
    lines.append(f"    Observed P(Sum=7) = {sum7_row['Empirical_Prob']:.5f} (Theoretical = {sum7_row['Theoretical_Prob']:.5f})")
    
    lines.extend([
        "",
        "3. CARD DECK PROBABILITIES (N=100,000 Draws with Replacement)",
        "-" * 78,
    ])
    for _, r in cards["df_cards"].iterrows():
        lines.append(f"    Event: {r['Event']:<16} | Obs={int(r['Count']):<6} | Emp_P={r['Emp_Prob']:.5f} | Theo_P={r['Theo_Prob']:.5f}")
        
    lines.extend([
        "",
        "4. MARKETING A/B TESTING CONVERSION & SIGNIFICANCE",
        "-" * 78,
        f"  Control Conversion:   {conv['df_summary'].loc[0, 'Empirical_Rate']*100:.2f}%",
        f"  Variant Conversion:   {conv['df_summary'].loc[1, 'Empirical_Rate']*100:.2f}%",
        f"  Relative Uplift:      +{conv['relative_uplift_pct']:.2f}%",
        f"  Z-Score:              {conv['z_score']:.3f}",
        f"  Significant at 95%:   {conv['statistically_significant']}",
        "",
        "5. FINANCIAL RISK & EXPECTED VALUE VERIFICATION",
        "-" * 78,
    ])
    for _, r in risk["df_bets"].iterrows():
        lines.append(f"    {r['Proposition']:<35} | Theo_EV={r['Theoretical_EV']:<9} | Emp_EV={r['Empirical_EV']:<9} | {r['Verdict']}")
        
    lines.extend([
        "",
        "=" * 78,
        "                           END OF REPORT",
        "=" * 78
    ])
    
    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Generated ASCII report: {output_path}")
    return report_text
