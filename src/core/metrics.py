"""
Evaluation Metrics.

Provides functions to compute classification performance and cross-entropy loss.
"""

import numpy as np


def evaluate(model, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Evaluates the prediction accuracy and cross-entropy loss of a trained model.

    Args:
        model (SoftmaxRegression or OneHiddenLayerNN): Trained classifier instance.
        features (np.ndarray): Input feature matrix of shape (n, d).
        labels (np.ndarray): Ground-truth class label array of shape (n,).

    Returns:
        tuple[float, float]: Cross-entropy loss and classification accuracy as a fraction.
    """
    cross_entropy_loss = model.compute_loss(features, labels)
    predicted_classes = model.predict(features)
    classification_accuracy = np.mean(predicted_classes == labels)
    return cross_entropy_loss, classification_accuracy
