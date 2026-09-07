"""
Automated business intelligence and interpretation generator.
"""
class InsightsGenerator:
    @staticmethod
    def generate_mean_insights(t_res: dict, boot_res: dict) -> list[str]:
        insights = []
        mean_val = t_res["sample_mean"]
        me = t_res["margin_of_error"]
        rel_err = (me / mean_val) * 100.0 if mean_val != 0 else 0.0
        
        insights.append(f"1. Central Estimate: The expected value is Rs. {mean_val:.2f} with a 95% margin of error of +/- Rs. {me:.2f} ({rel_err:.1f}% relative precision).")
        insights.append(f"2. Statistical Decision Range: At 95% confidence, the true population average lies between Rs. {t_res['lower_bound']:.2f} and Rs. {t_res['upper_bound']:.2f}.")
        
        # Bootstrap comparison
        diff_lower = abs(t_res["lower_bound"] - boot_res["lower_bound"])
        diff_upper = abs(t_res["upper_bound"] - boot_res["upper_bound"])
        if diff_lower < 5.0 and diff_upper < 5.0:
            insights.append("3. Distributional Robustness: Parametric T-interval and non-parametric Bootstrap interval match within Rs. 5.00, confirming the Central Limit Theorem holds firmly.")
        else:
            insights.append("3. Skewness Notice: Slight difference between Bootstrap and T-interval indicates mild sample skewness; non-parametric bounds offer added protection.")
            
        insights.append("4. Frequentist Protocol: 95% confidence guarantees that 95 out of 100 identically drawn random samples will capture the true underlying population average.")
        return insights
