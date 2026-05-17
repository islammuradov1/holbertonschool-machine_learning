#!/usr/bin/env python3
"""Q affinities for t-SNE"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities for t-SNE.

    Parameters:
        Y (numpy.ndarray): shape (n, ndim) - low dimensional transformation

    Returns:
        Q (numpy.ndarray): shape (n, n) - Q affinities
        num (numpy.ndarray): shape (n, n) - numerator of Q affinities
    """
    # Squared pairwise distances in low-dimensional space
    sum_Y = np.sum(Y ** 2, axis=1)
    D = sum_Y + sum_Y[:, np.newaxis] - 2 * np.dot(Y, Y.T)

    # Numerator: (1 + ||yi - yj||^2)^-1  (Student t-distribution kernel)
    num = 1 / (1 + D)

    # Set diagonal to 0
    np.fill_diagonal(num, 0)

    # Normalize to get Q
    Q = num / np.sum(num)

    return Q, num
