"""
Day 72 — Challenge 2: Simpson's Paradox Demonstration
Demonstrates subgroup trend reversal where Strategy B outperforms Strategy A in every device segment,
yet Strategy A wins overall due to disparate segment allocation.
"""
import numpy as np
import pandas as pd

def main():
    # Subgroup 1: Desktop Users
    # Strategy A: 800 visitors, 160 conversions (20.0%)
    # Strategy B: 200 visitors, 60 conversions  (30.0%) -> B wins!
    
    # Subgroup 2: Mobile Users
    # Strategy A: 200 visitors, 16 conversions  (8.0%)
    # Strategy B: 800 visitors, 96 conversions  (12.0%) -> B wins!
    
    records = []
    # Desktop A
    for _ in range(160): records.append({"Device": "Desktop", "Strategy": "A", "Converted": 1})
    for _ in range(800 - 160): records.append({"Device": "Desktop", "Strategy": "A", "Converted": 0})
    # Desktop B
    for _ in range(60): records.append({"Device": "Desktop", "Strategy": "B", "Converted": 1})
    for _ in range(200 - 60): records.append({"Device": "Desktop", "Strategy": "B", "Converted": 0})
    
    # Mobile A
    for _ in range(16): records.append({"Device": "Mobile", "Strategy": "A", "Converted": 1})
    for _ in range(200 - 16): records.append({"Device": "Mobile", "Strategy": "A", "Converted": 0})
    # Mobile B
    for _ in range(96): records.append({"Device": "Mobile", "Strategy": "B", "Converted": 1})
    for _ in range(800 - 96): records.append({"Device": "Mobile", "Strategy": "B", "Converted": 0})
    
    df = pd.DataFrame(records)
    
    # 1. Segmented Analysis
    subgroup = df.groupby(["Device", "Strategy"])["Converted"].agg(["count", "mean"]).reset_index()
    subgroup["Conversion_Rate"] = (subgroup["mean"] * 100).round(1).astype(str) + "%"
    
    # 2. Aggregated Analysis
    aggregate = df.groupby("Strategy")["Converted"].agg(["count", "mean"]).reset_index()
    aggregate["Conversion_Rate"] = (aggregate["mean"] * 100).round(1).astype(str) + "%"
    
    print("=" * 65)
    print("DAY 72 — CHALLENGE 2: SIMPSON'S PARADOX")
    print("=" * 65)
    print("Subgroup Conversion Rates (Stratified by Device):")
    print(subgroup[["Device", "Strategy", "count", "Conversion_Rate"]].to_string(index=False))
    print("\nAggregate Conversion Rates (Combined Population):")
    print(aggregate[["Strategy", "count", "Conversion_Rate"]].to_string(index=False))
    
    cr_a_desk = df[(df["Device"]=="Desktop") & (df["Strategy"]=="A")]["Converted"].mean()
    cr_b_desk = df[(df["Device"]=="Desktop") & (df["Strategy"]=="B")]["Converted"].mean()
    cr_a_mob = df[(df["Device"]=="Mobile") & (df["Strategy"]=="A")]["Converted"].mean()
    cr_b_mob = df[(df["Device"]=="Mobile") & (df["Strategy"]=="B")]["Converted"].mean()
    cr_a_all = df[df["Strategy"]=="A"]["Converted"].mean()
    cr_b_all = df[df["Strategy"]=="B"]["Converted"].mean()
    
    assert cr_b_desk > cr_a_desk, "Strategy B must win on Desktop!"
    assert cr_b_mob > cr_a_mob, "Strategy B must win on Mobile!"
    assert cr_a_all > cr_b_all, "Strategy A must appear to win in aggregate!"
    
    print("\nParadox Explanation:")
    print(f"  Desktop:  B ({cr_b_desk*100:.1f}%) > A ({cr_a_desk*100:.1f}%)  --> B wins (+10.0%)")
    print(f"  Mobile:   B ({cr_b_mob*100:.1f}%) > A ({cr_a_mob*100:.1f}%)  --> B wins (+4.0%)")
    print(f"  Overall:  A ({cr_a_all*100:.1f}%) > B ({cr_b_all*100:.1f}%)  --> A wins (+2.0%) [PARADOX]")
    print("  Root Cause: 80% of Strategy A users were on high-converting Desktop, while")
    print("  80% of Strategy B users were assigned to low-converting Mobile.")
    print("  Takeaway: Never evaluate conversion without controlling for cohort allocations!")

if __name__ == "__main__":
    main()
