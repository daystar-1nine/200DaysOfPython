"""
Bayesian inference and probability updating algorithms.
"""

def bayes_theorem(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """
    Apply Bayes' Rule:
    P(A | B) = [P(B | A) * P(A)] / P(B)
    """
    if not (0.0 <= p_b_given_a <= 1.0 and 0.0 <= p_a <= 1.0 and 0.0 < p_b <= 1.0):
        raise ValueError("Probabilities must be valid; p_b must be > 0.")
    posterior = (p_b_given_a * p_a) / p_b
    return min(1.0, max(0.0, posterior))

def bayes_update(
    prior: float,
    likelihood_true: float,
    likelihood_false: float
) -> dict:
    """
    Compute binary Bayesian update:
    P(Hypothesis | Evidence) = [P(E|H) * P(H)] / [P(E|H)*P(H) + P(E|~H)*P(~H)]
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood_true <= 1.0 and 0.0 <= likelihood_false <= 1.0):
        raise ValueError("All probabilities must be in [0, 1].")
    
    prior_not_h = 1.0 - prior
    marginal_evidence = (likelihood_true * prior) + (likelihood_false * prior_not_h)
    
    if marginal_evidence == 0:
        posterior = 0.0
    else:
        posterior = (likelihood_true * prior) / marginal_evidence
        
    return {
        "prior": prior,
        "likelihood_true": likelihood_true,
        "likelihood_false": likelihood_false,
        "marginal_evidence": round(marginal_evidence, 6),
        "posterior": round(posterior, 6)
    }
