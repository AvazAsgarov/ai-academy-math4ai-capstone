"""
Neural Network Models.

Implements the One-Hidden-Layer Neural Network architecture.
"""

import numpy as np

from src.math_ops.utilities import compute_stable_softmax


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

        self.optimizer_type = 'sgd'
        self.timestep = 0
        self.velocity_storage = {}
        self.moment_storage = {}

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
        Applies gradient descent to update all network parameters using the configured optimizer.

        Args:
            gradient_weight_1 (np.ndarray): Hidden-layer weight gradient of shape (h, d).
            gradient_bias_1 (np.ndarray): Hidden-layer bias gradient of shape (h,).
            gradient_weight_2 (np.ndarray): Output-layer weight gradient of shape (k, h).
            gradient_bias_2 (np.ndarray): Output-layer bias gradient of shape (k,).
            learning_rate (float): Step size for the gradient descent update.
        """
        gradients = {
            'w1': gradient_weight_1,
            'b1': gradient_bias_1,
            'w2': gradient_weight_2,
            'b2': gradient_bias_2
        }

        parameters = {
            'w1': self.weight_matrix_1,
            'b1': self.bias_vector_1,
            'w2': self.weight_matrix_2,
            'b2': self.bias_vector_2
        }

        if not self.velocity_storage:
            for key, param in parameters.items():
                self.velocity_storage[key] = np.zeros_like(param)
                self.moment_storage[key] = np.zeros_like(param)

        self.timestep += 1

        for key in parameters:
            grad = gradients[key]
            
            if self.optimizer_type == 'sgd':
                parameters[key] -= learning_rate * grad

            elif self.optimizer_type == 'momentum':
                beta = 0.9
                self.velocity_storage[key] = beta * self.velocity_storage[key] + (1 - beta) * grad
                parameters[key] -= learning_rate * self.velocity_storage[key]

            elif self.optimizer_type == 'adam':
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8
                
                self.moment_storage[key] = beta1 * self.moment_storage[key] + (1 - beta1) * grad
                self.velocity_storage[key] = beta2 * self.velocity_storage[key] + (1 - beta2) * (grad ** 2)
                
                m_hat = self.moment_storage[key] / (1 - beta1 ** self.timestep)
                v_hat = self.velocity_storage[key] / (1 - beta2 ** self.timestep)
                
                parameters[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        self.weight_matrix_1 = parameters['w1']
        self.bias_vector_1 = parameters['b1']
        self.weight_matrix_2 = parameters['w2']
        self.bias_vector_2 = parameters['b2']
