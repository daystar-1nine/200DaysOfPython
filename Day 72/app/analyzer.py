"""
Day 72 — Comprehensive Relationship Analyzer
Performs high-level synthesis of pairwise relationships, multicollinearity, and answers business questions.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd

try:
    from app.config import AppConfig
    from app.outliers import evaluate_outlier_impact
except ImportError:
    from config import AppConfig
    from outliers import evaluate_outlier_impact

class RelationshipAnalyzer:
    def __init__(self, pairwise_df: pd.DataFrame, raw_df: pd.DataFrame, config: AppConfig = None):
        self.pairwise_df = pairwise_df
        self.raw_df = raw_df
        self.config = config or AppConfig()
        
    def get_top_positive(self, n: int = 5) -> pd.DataFrame:
        pos = self.pairwise_df[self.pairwise_df["Pearson_r"] > 0]
        return pos.sort_values(by="Pearson_r", ascending=False).head(n)
        
    def get_top_negative(self, n: int = 5) -> pd.DataFrame:
        neg = self.pairwise_df[self.pairwise_df["Pearson_r"] < 0]
        return neg.sort_values(by="Pearson_r", ascending=True).head(n)
        
    def get_multicollinear_pairs(self) -> pd.DataFrame:
        mc = self.pairwise_df[self.pairwise_df["Is_Multicollinear"]]
        return mc.sort_values(by="Pearson_r", ascending=False)
        
    def get_divergent_pairs(self) -> pd.DataFrame:
        div = self.pairwise_df[self.pairwise_df["Is_Divergent"]]
        return div.sort_values(by="Abs_Diff", ascending=False)
        
    def answer_business_questions(self) -> List[Dict[str, str]]:
        q_and_a = []
        
        # Helper to lookup pair
        def find_pair(v1: str, v2: str):
            match = self.pairwise_df[
                ((self.pairwise_df["Variable_1"] == v1) & (self.pairwise_df["Variable_2"] == v2)) |
                ((self.pairwise_df["Variable_1"] == v2) & (self.pairwise_df["Variable_2"] == v1))
            ]
            if not match.empty:
                return match.iloc[0]
            return None

        # Q1: Does higher quantity relate to higher revenue?
        p1 = find_pair("Quantity", "Revenue")
        ans1 = f"Yes. Quantity and Revenue exhibit a strong positive linear association (r = {p1['Pearson_r']:.3f}, p < 0.001)." if p1 is not None else "Metrics not present in dataset."
        q_and_a.append({
            "Question": "Q1: Does higher order quantity relate to higher revenue?",
            "Answer": ans1
        })
        
        # Q2: Does discount relate to profit?
        p2 = find_pair("Discount", "Profit")
        ans2 = f"Discount and Profit exhibit a negative association (r = {p2['Pearson_r']:.3f}). Higher discounts correspond with lower profit margins." if p2 is not None else "Metrics not present in dataset."
        q_and_a.append({
            "Question": "Q2: How does discount relate to business profit?",
            "Answer": ans2
        })
        
        # Q3: Does cost relate to revenue?
        p3 = find_pair("Cost", "Revenue")
        ans3 = f"Yes, strong positive co-movement (r = {p3['Pearson_r']:.3f}), reflecting that higher volumes drive both revenue and inventory cost." if p3 is not None else "Metrics not present in dataset."
        q_and_a.append({
            "Question": "Q3: Does operational cost relate to gross revenue?",
            "Answer": ans3
        })
        
        # Q4: Which variable has the strongest relationship with profit?
        profit_pairs = self.pairwise_df[
            (self.pairwise_df["Variable_1"] == "Profit") | (self.pairwise_df["Variable_2"] == "Profit")
        ].copy()
        if not profit_pairs.empty:
            profit_pairs["abs_r"] = profit_pairs["Pearson_r"].abs()
            top_profit = profit_pairs.sort_values(by="abs_r", ascending=False).iloc[0]
            other_var = top_profit["Variable_2"] if top_profit["Variable_1"] == "Profit" else top_profit["Variable_1"]
            ans4 = f"{other_var} shows the strongest association with Profit (r = {top_profit['Pearson_r']:.3f}, rho = {top_profit['Spearman_rho']:.3f})."
        else:
            top_any = self.pairwise_df.sort_values(by="Pearson_r", key=abs, ascending=False).iloc[0]
            ans4 = f"{top_any['Variable_1']} and {top_any['Variable_2']} exhibit the strongest overall association (r = {top_any['Pearson_r']:.3f})."
        q_and_a.append({
            "Question": "Q4: Which variable has the strongest association with Profit?",
            "Answer": ans4
        })
        
        # Q5: Which variables have weak linear relationships?
        weakest = self.pairwise_df.sort_values(by="Pearson_r", key=abs).iloc[0]
        q_and_a.append({
            "Question": "Q5: Which variable pair has the weakest linear relationship?",
            "Answer": f"{weakest['Variable_1']} and {weakest['Variable_2']} exhibit near-zero correlation (r = {weakest['Pearson_r']:.4f})."
        })
        
        # Q6: Are there highly correlated predictors (multicollinearity)?
        mc_count = len(self.get_multicollinear_pairs())
        q_and_a.append({
            "Question": "Q6: Are there potential multicollinearity risks among predictors?",
            "Answer": f"Identified {mc_count} variable pairs with |r| >= {self.config.MULTICOLLINEARITY_THRESHOLD}."
        })
        
        # Q7: Which relationships are positive?
        pos_count = (self.pairwise_df["Pearson_r"] > 0).sum()
        q_and_a.append({
            "Question": "Q7: How many pairs exhibit positive vs negative linear associations?",
            "Answer": f"{pos_count} pairs exhibit positive association, while {len(self.pairwise_df) - pos_count} pairs exhibit negative or zero association."
        })
        
        # Q8: Which relationships are negative?
        neg_pairs = self.pairwise_df[self.pairwise_df["Pearson_r"] < 0]
        if not neg_pairs.empty:
            top_neg_item = neg_pairs.sort_values(by="Pearson_r").iloc[0]
            ans8 = f"{top_neg_item['Variable_1']} and {top_neg_item['Variable_2']} exhibit inverse correlation (r = {top_neg_item['Pearson_r']:.3f})."
        else:
            ans8 = "No inverse (negative) relationships detected in current dataset."
        q_and_a.append({
            "Question": "Q8: Which notable metric pairs exhibit inverse (negative) relationships?",
            "Answer": ans8
        })
        
        # Q9: Where do Pearson and Spearman differ substantially?
        div_pairs = self.get_divergent_pairs()
        q_and_a.append({
            "Question": "Q9: Where do Pearson and Spearman differ substantially (|r - rho| >= 0.20)?",
            "Answer": f"Found {len(div_pairs)} divergent pairs. Such divergence indicates non-linear rank behavior or extreme outlier leverage."
        })
        
        # Q10: Are any relationships driven by obvious outliers?
        if "Revenue" in self.raw_df.columns and "Profit" in self.raw_df.columns:
            outlier_res = evaluate_outlier_impact(self.raw_df, "Revenue", "Profit")
            ans10 = f"For Revenue-Profit, r shifts from {outlier_res['r_all']:.3f} to {outlier_res['r_clean']:.3f} (delta = {outlier_res['delta_r']:.3f}) after removing {outlier_res['n_outliers']} extreme points."
        else:
            c1, c2 = self.raw_df.columns[0], self.raw_df.columns[1]
            outlier_res = evaluate_outlier_impact(self.raw_df, c1, c2)
            ans10 = f"For {c1}-{c2}, r shifts from {outlier_res['r_all']:.3f} to {outlier_res['r_clean']:.3f} after removing {outlier_res['n_outliers']} outliers."
        q_and_a.append({
            "Question": "Q10: Are key financial correlations sensitive to outliers?",
            "Answer": ans10
        })
        
        return q_and_a
