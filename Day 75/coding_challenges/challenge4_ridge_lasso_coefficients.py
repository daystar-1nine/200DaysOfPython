import numpy as np
def count_zero_coefficients(coefs):
    return np.sum(np.isclose(coefs, 0))
