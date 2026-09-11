"""
Task 6: Cohen's d Effect Size Calculation
Measures practical effect magnitude independent of sample size.
"""

def calculate_cohens_d(sample_mean: float, mu_0: float, sample_std: float) -> dict:
    if sample_std <= 0:
        raise ValueError("Sample standard deviation must be positive.")
    
    diff = sample_mean - mu_0
    d = diff / sample_std
    abs_d = abs(d)
    
    if abs_d < 0.20:
        interpretation = "Negligible"
    elif abs_d < 0.50:
        interpretation = "Small"
    elif abs_d < 0.80:
        interpretation = "Medium"
    else:
        interpretation = "Large"
        
    return {
        "sample_mean": round(sample_mean, 4),
        "hypothesized_mean": round(mu_0, 4),
        "mean_difference": round(diff, 4),
        "sample_std": round(sample_std, 4),
        "cohens_d": round(d, 4),
        "effect_size_magnitude": interpretation
    }

if __name__ == "__main__":
    cases = [
        {"x_bar": 101.0, "mu_0": 100.0, "s": 15.0},  # Small difference, large std
        {"x_bar": 108.0, "mu_0": 100.0, "s": 15.0},  # Medium difference
        {"x_bar": 120.0, "mu_0": 100.0, "s": 15.0},  # Large difference
    ]
    print("=== Task 6: Cohen's d Effect Sizes ===")
    for c in cases:
        out = calculate_cohens_d(c["x_bar"], c["mu_0"], c["s"])
        print(f"  Diff={out['mean_difference']} | s={out['sample_std']} -> d={out['cohens_d']} ({out['effect_size_magnitude']})")
