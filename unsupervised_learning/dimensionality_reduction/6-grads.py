#!/usr/bin/env python3
"""Gradients of Y for t-SNE"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Parameters:
        Y (numpy.ndarray): shape (n, ndim) - low dimensional transformation
        P (numpy.ndarray): shape (n, n) - P affinities of X

    Returns:
        dY (numpy.ndarray): shape (n, ndim) - gradients of Y
        Q (numpy.ndarray): shape (n, n) - Q affinities of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    # (P - Q) * num: element-wise, shape (n, n)
    PQ = (P - Q) * num

    # Gradient: sum over j of (P_ij - Q_ij) * num_ij * (yi - yj)
    dY = np.zeros((n, ndim))
    for i in range(n):
        # (n, 1) * (n, ndim) -> sum -> (ndim,)
        dY[i] = np.sum(PQ[i, :, np.newaxis] * (Y[i] - Y), axis=0)

    return dY, Q
