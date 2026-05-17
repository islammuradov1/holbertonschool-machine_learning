#!/usr/bin/env python3
"""Cost function for t-SNE"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation (KL divergence).

    Parameters:
        P (numpy.ndarray): shape (n, n) - P affinities
        Q (numpy.ndarray): shape (n, n) - Q affinities

    Returns:
        C (float): cost of the transformation
    """
    # Avoid division by zero / log(0)
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    # KL divergence: sum over all i,j of P_ij * log(P_ij / Q_ij)
    C = np.sum(P * np.log(P / Q))

    return C
