import numpy as np

def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining x successes in n trials
    for each probability in P (Binomial distribution).
    """

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient (n choose x)
    coeff = np.math.comb(n, x)

    # Likelihood calculation
    likelihoods = coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
