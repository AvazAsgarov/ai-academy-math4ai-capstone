"""
Mathematical Utilities.

Provides low-level vectorized mathematical operations required
for probability distributions and stability.
"""

import numpy as np


def compute_stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Computes numerically stable softmax probabilities across a batch of logits.

    Args:
        logits (np.ndarray): Raw unnormalized score matrix of shape (n, k),
            where n is the batch size and k is the number of classes.

    Returns:
        np.ndarray: Probability matrix of shape (n, k) where each row sums to 1.
    """
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated_logits = np.exp(shifted_logits)
    return exponentiated_logits / np.sum(exponentiated_logits, axis=1, keepdims=True)
