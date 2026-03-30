"""
Neural Network and Regression Models.

Implements Softmax Regression and a One-Hidden-Layer Neural Network
from scratch using NumPy, optimized for vector operations.
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


class OneHiddenLayerNN:
    """
    Multilayer perceptron consisting of a single hidden layer with Tanh activation.
    """

    def __init__(
        self,
        input_dimensions: int,
        hidden_dimensions: int,
        total_classes: int,
        l2_regularization: float = 1e-4
    ) -> None:
        """
        Configures the layer connections utilizing Glorot uniform initialization matrices.

        Args:
            input_dimensions (int): Dimensionality of each input feature vector.
            hidden_dimensions (int): Number of neurons in the hidden layer.
            total_classes (int): Number of target classification categories.
            l2_regularization (float): L2 penalty coefficient for weight decay.
        """
        variance_limit_1 = np.sqrt(6.0 / (input_dimensions + hidden_dimensions))
        self.weight_matrix_1 = np.random.uniform(-variance_limit_1, variance_limit_1, (hidden_dimensions, input_dimensions))
        self.bias_vector_1 = np.zeros(hidden_dimensions)

        variance_limit_2 = np.sqrt(6.0 / (hidden_dimensions + total_classes))
        self.weight_matrix_2 = np.random.uniform(-variance_limit_2, variance_limit_2, (total_classes, hidden_dimensions))
        self.bias_vector_2 = np.zeros(total_classes)

        self.l2_regularization = l2_regularization

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagates features through the hidden layer to produce output logits.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Pre-activation hidden values (n, h),
                activated hidden values (n, h), and output logits (n, k).
        """
        pre_activation_hidden = features.dot(self.weight_matrix_1.T) + self.bias_vector_1
        activated_hidden = np.tanh(pre_activation_hidden)
        logits = activated_hidden.dot(self.weight_matrix_2.T) + self.bias_vector_2

        return pre_activation_hidden, activated_hidden, logits

    def predict_probs(self, features: np.ndarray) -> np.ndarray:
        """
        Transforms network logits into normalized categorical probability distributions.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).

        Returns:
            np.ndarray: Probability matrix of shape (n, k).
        """
        _, _, logits = self.forward(features)
        return compute_stable_softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Determines the class label with the highest predicted probability.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).

        Returns:
            np.ndarray: Predicted class label array of shape (n,).
        """
        probabilities = self.predict_probs(features)
        return np.argmax(probabilities, axis=1)

    def compute_loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Measures L2-regularized sparse categorical Cross-Entropy across both layers.

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

        regularization_term_1 = np.sum(self.weight_matrix_1 ** 2)
        regularization_term_2 = np.sum(self.weight_matrix_2 ** 2)
        total_regularization = 0.5 * self.l2_regularization * (regularization_term_1 + regularization_term_2)

        return core_loss + total_regularization

    def backward(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagates the loss to compute gradients for all layer parameters.

        Args:
            features (np.ndarray): Input feature matrix of shape (n, d).
            labels (np.ndarray): Ground-truth class label array of shape (n,).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Gradients for
                weight_1 (h, d), bias_1 (h,), weight_2 (k, h), and bias_2 (k,).
        """
        total_samples = features.shape[0]
        _, activated_hidden, logits = self.forward(features)
        probabilities = compute_stable_softmax(logits)

        gradient_output_scores = probabilities.copy()
        gradient_output_scores[np.arange(total_samples), labels] -= 1
        gradient_output_scores /= total_samples

        gradient_weight_2 = gradient_output_scores.T.dot(activated_hidden) + self.l2_regularization * self.weight_matrix_2
        gradient_bias_2 = np.sum(gradient_output_scores, axis=0)

        gradient_hidden_activations = gradient_output_scores.dot(self.weight_matrix_2) * (1 - activated_hidden**2)
        gradient_weight_1 = gradient_hidden_activations.T.dot(features) + self.l2_regularization * self.weight_matrix_1
        gradient_bias_1 = np.sum(gradient_hidden_activations, axis=0)

        return gradient_weight_1, gradient_bias_1, gradient_weight_2, gradient_bias_2

    def update(
        self,
        gradient_weight_1: np.ndarray,
        gradient_bias_1: np.ndarray,
        gradient_weight_2: np.ndarray,
        gradient_bias_2: np.ndarray,
        learning_rate: float
    ) -> None:
        """
        Applies gradient descent to update all network parameters.

        Args:
            gradient_weight_1 (np.ndarray): Hidden-layer weight gradient of shape (h, d).
            gradient_bias_1 (np.ndarray): Hidden-layer bias gradient of shape (h,).
            gradient_weight_2 (np.ndarray): Output-layer weight gradient of shape (k, h).
            gradient_bias_2 (np.ndarray): Output-layer bias gradient of shape (k,).
            learning_rate (float): Step size for the gradient descent update.
        """
        self.weight_matrix_1 -= learning_rate * gradient_weight_1
        self.bias_vector_1 -= learning_rate * gradient_bias_1
        self.weight_matrix_2 -= learning_rate * gradient_weight_2
        self.bias_vector_2 -= learning_rate * gradient_bias_2
