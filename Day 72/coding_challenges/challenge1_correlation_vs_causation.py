"""
Day 72 — Challenge 1: Correlation vs Causation & Confounding Simulation
Simulates common cause confounding (Temperature -> Ice Cream & Drowning)
and computes partial correlation rho(XY.Z) to prove absence of direct causal link.
"""
import numpy as np
from scipy import stats

def partial_correlation(r_xy: float, r_xz: float, r_yz: float) -> float:
    denom = np.sqrt(1.0 - r_xz ** 2) * np.sqrt(1.0 - r_yz ** 2)
    if denom == 0:
        return 0.0
    return float((r_xy - r_xz * r_yz) / denom)

def main():
    np.random.seed(42)
    n = 1000
    
    # Confounder Z: Average Daily Temperature (Celsius)
    temp_z = np.random.normal(loc=28.0, scale=6.0, size=n)
    
    # Variable X: Daily Ice Cream Sales (Rupees in thousands)
    ice_cream_x = 15.0 + 3.2 * temp_z + np.random.normal(loc=0.0, scale=8.0, size=n)
    
    # Variable Y: Daily Drowning Incidents
    drowning_y = 0.5 + 0.35 * temp_z + np.random.normal(loc=0.0, scale=1.2, size=n)
    drowning_y = np.clip(drowning_y, 0, None)
    
    r_xy, p_xy = stats.pearsonr(ice_cream_x, drowning_y)
    r_xz, p_xz = stats.pearsonr(ice_cream_x, temp_z)
    r_yz, p_yz = stats.pearsonr(drowning_y, temp_z)
    
    r_partial = partial_correlation(r_xy, r_xz, r_yz)
    
    print("=" * 65)
    print("DAY 72 — CHALLENGE 1: CORRELATION VS CAUSATION & CONFOUNDING")
    print("=" * 65)
    print(f"Sample size N = {n}")
    print(f"Observed Zero-Order Correlations:")
    print(f"  r(Ice Cream, Drowning)    = {r_xy:.4f} (p = {p_xy:.4e}) [Strong Positive]")
    print(f"  r(Ice Cream, Temperature) = {r_xz:.4f} (p = {p_xz:.4e})")
    print(f"  r(Drowning, Temperature)  = {r_yz:.4f} (p = {p_yz:.4e})")
    print("\nControlling for Confounder (Temperature):")
    print(f"  Partial Correlation rho(Ice Cream, Drowning | Temperature) = {r_partial:.4f}")
    
    assert r_xy > 0.70, "Expected strong spurious zero-order correlation!"
    assert abs(r_partial) < 0.10, "Partial correlation should drop near zero!"
    print("\nStatistical Deduction:")
    print("  The zero-order correlation (r = {:.2f}) is spurious, driven entirely by".format(r_xy))
    print("  the common confounding cause (Temperature). Once Temperature is controlled for,")
    print("  the partial correlation collapses to near zero ({:.4f}).".format(r_partial))
    print("  Conclusion: Ice cream consumption does NOT cause drowning!")

if __name__ == "__main__":
    main()
