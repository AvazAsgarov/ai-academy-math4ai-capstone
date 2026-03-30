"""
Linear Regression Models.

Implements the baseline Softmax Regression architecture.
"""

import numpy as np

from src.math_ops.utilities import compute_stable_softmax


class SoftmaxRegression:
    """
    Linear baseline model for multiclass classification using Cross-Entropy loss.
    """

    def __init__(self, input_dimensions: int, total_classes: int, l2_regularization: float = 1e-4) -> None:
        """
        Initializes weights using Xavier-like scaling and sets biases to zero.

        Args:
            input_dimensions (int): Dimensionality of each input feature vector.
            total_classes (int): Number of target classification categories.
            l2_regularization (float): L2 penalty coefficient for weight decay.
        """
        self.weight_matrix = np.random.randn(total_classes, input_dimensions) * np.sqrt(1.0 / input_dimensions)
        self.bias_vector = np.zeros(total_classes)
        self.l2_regularization = l2_regularization

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Executes the linear forward pass computing raw unnormalized scores.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).

        Returns:
            np.ndarray: Logit score matrix of shape (n, k).
        """
        return features.dot(self.weight_matrix.T) + self.bias_vector

    def predict_probs(self, features: np.ndarray) -> np.ndarray:
        """
        Applies softmax to the linear activations yielding probability distributions.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).

        Returns:
            np.ndarray: Probability matrix of shape (n, k).
        """
        logits = self.forward(features)
        return compute_stable_softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Infers the highest-probability class assignment for each input sample.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).

        Returns:
            np.ndarray: Predicted class label array of shape (n,).
        """
        probabilities = self.predict_probs(features)
        return np.argmax(probabilities, axis=1)

    def compute_loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Evaluates the L2-regularized Cross-Entropy loss for a complete batch.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).
            labels (np.ndarray): Ground-truth class label array of shape (n,).

        Returns:
            float: Scalar loss value combining cross-entropy and regularization.
        """
        total_samples = features.shape[0]
        probabilities = self.predict_probs(features)

        predicted_correct_class = probabilities[np.arange(total_samples), labels]
        core_loss = -np.sum(np.log(predicted_correct_class + 1e-15)) / total_samples
        regularization_loss = 0.5 * self.l2_regularization * np.sum(self.weight_matrix ** 2)

        return core_loss + regularization_loss

    def backward(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Derives the gradients of the loss with respect to weights and biases.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).
            labels (np.ndarray): Ground-truth class label array of shape (n,).

        Returns:
            tuple[np.ndarray, np.ndarray]: Gradient of weights (k, d) and gradient of biases (k,).
        """
        total_samples = features.shape[0]
        probabilities = self.predict_probs(features)

        gradient_scores = probabilities.copy()
        gradient_scores[np.arange(total_samples), labels] -= 1
        gradient_scores /= total_samples

        gradient_weights = gradient_scores.T.dot(features) + self.l2_regularization * self.weight_matrix
        gradient_biases = np.sum(gradient_scores, axis=0)

        return gradient_weights, gradient_biases

    def update(self, gradient_weights: np.ndarray, gradient_biases: np.ndarray, learning_rate: float) -> None:
        """
        Performs a steepest descent update integrating the computed gradients.

        Args:
            gradient_weights (np.ndarray): Weight gradient matrix of shape (k, d).
            gradient_biases (np.ndarray): Bias gradient vector of shape (k,).
            learning_rate (float): Step size for the gradient descent update.
        """
        self.weight_matrix -= learning_rate * gradient_weights
        self.bias_vector -= learning_rate * gradient_biases
