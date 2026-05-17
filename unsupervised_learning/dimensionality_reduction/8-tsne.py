#!/usr/bin/env python3
"""t-SNE transformation"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation on a dataset.

    Parameters:
        X (numpy.ndarray): shape (n, d) - dataset to transform
        ndims (int): new dimensional representation
        idims (int): intermediate PCA dimensions
        perplexity (float): perplexity for t-SNE
        iterations (int): number of iterations
        lr (float): learning rate

    Returns:
        Y (numpy.ndarray): shape (n, ndims) - optimized low-dim transformation
    """
    n, d = X.shape

    # Step 1: PCA reduction to idims
    X = pca(X, idims)

    # Step 2: Compute P affinities
    P = P_affinities(X, perplexity=perplexity)

    # Step 3: Initialize Y randomly
    Y = np.random.randn(n, ndims)

    # Initialize momentum variables
    iY = np.zeros((n, ndims))   # previous update (Y(t-1) - Y(t-2))

    # Early exaggeration for first 100 iterations
    P_exag = P * 4

    for t in range(1, iterations + 1):
        # Use exaggerated P for first 100 iterations
        if t <= 100:
            P_use = P_exag
        else:
            P_use = P

        # Momentum
        if t <= 20:
            alpha = 0.5
        else:
            alpha = 0.8

        # Compute gradients (without the factor of 4)
        dY, Q = grads(Y, P_use)

        # Gradient descent with momentum
        # Correct update: Y(t) = Y(t-1) + lr * grad + alpha * (Y(t-1) - Y(t-2))
        iY = alpha * iY - lr * dY
        Y = Y + iY

        # Re-center Y
        Y = Y - np.mean(Y, axis=0)

        # Print cost every 100 iterations
        if t % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(t, C))

    return Y
