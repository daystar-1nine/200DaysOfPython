"""
Day 66 Coding Challenge 4: Bayesian Customer Churn & Diagnostic Sensitivity Engine
Computes posterior probability of customer churn given inactivity indicators,
and runs sensitivity analysis across varying base priors and false alarm rates.
"""
import pandas as pd

def calculate_bayes_churn(
    p_churn_prior: float,
    p_inactive_given_churn: float,
    p_inactive_given_active: float
) -> dict:
    """
    Compute P(Churn | Inactive) via Bayes' Theorem:
    P(Churn | I) = P(I | Churn) * P(Churn) / P(I)
    where P(I) = P(I | Churn)*P(Churn) + P(I | Active)*P(Active)
    """
    p_active_prior = 1.0 - p_churn_prior
    p_evidence_inactive = (p_inactive_given_churn * p_churn_prior) + (p_inactive_given_active * p_active_prior)
    
    if p_evidence_inactive == 0:
        return {"posterior": 0.0, "evidence": 0.0, "bayes_factor": 0.0}
        
    p_churn_given_inactive = (p_inactive_given_churn * p_churn_prior) / p_evidence_inactive
    bayes_factor = p_inactive_given_churn / p_inactive_given_active if p_inactive_given_active > 0 else float("inf")
    
    return {
        "prior_churn": p_churn_prior,
        "sensitivity": p_inactive_given_churn,
        "false_positive_rate": p_inactive_given_active,
        "total_inactive_rate": round(p_evidence_inactive, 4),
        "posterior_churn": round(p_churn_given_inactive, 4),
        "bayes_factor": round(bayes_factor, 2)
    }

def run_sensitivity_matrix(
    priors: list[float],
    false_pos_rates: list[float],
    sensitivity: float = 0.85
) -> pd.DataFrame:
    """Generate sensitivity matrix varying priors against false positive rates."""
    table = []
    for prior in priors:
        row = {"Prior P(Churn)": f"{prior*100:.1f}%"}
        for fpr in false_pos_rates:
            res = calculate_bayes_churn(prior, sensitivity, fpr)
            row[f"FPR={fpr*100:.0f}%"] = f"{res['posterior_churn']*100:.1f}%"
        table.append(row)
    return pd.DataFrame(table)

def main():
    print("=" * 70)
    print("  CHALLENGE 4: BAYESIAN CUSTOMER CHURN SENSITIVITY ENGINE")
    print("=" * 70)
    
    # Baseline Scenario
    prior = 0.12            # 12% customer churn base rate
    sensitivity = 0.80      # 80% of churned customers are inactive for >30d
    fpr = 0.15              # 15% of non-churned customers are also inactive
    
    base_res = calculate_bayes_churn(prior, sensitivity, fpr)
    print(f"Base Churn Rate:        {base_res['prior_churn']*100:.1f}%")
    print(f"Signal Sensitivity:     {base_res['sensitivity']*100:.1f}%")
    print(f"False Alarm Rate (FPR): {base_res['false_positive_rate']*100:.1f}%")
    print(f"Likelihood Ratio (BF):  {base_res['bayes_factor']}")
    print(f"--> Posterior P(Churn|Inactive): {base_res['posterior_churn']*100:.2f}%\n")
    
    # Sensitivity Analysis across prior frequencies and false alarms
    priors = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    fprs = [0.05, 0.10, 0.15, 0.25, 0.40]
    
    sens_df = run_sensitivity_matrix(priors, fprs, sensitivity=0.80)
    print("[Sensitivity Matrix: Posterior P(Churn | Inactive) with Sensitivity=80%]")
    print(sens_df.to_string(index=False))

if __name__ == "__main__":
    main()
