"""
Model Training Engine.

Provides the core training loop for batch gradient descent with 
optional validation set tracking.
"""

import numpy as np

from src.core.metrics import evaluate


def train_model(
    model,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray = None,
    labels_val: np.ndarray = None,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    return_history: bool = False,
    use_best_validation: bool = True
) -> dict[str, list[float]] | None:
    """
    Executes the mini-batch gradient descent training loop over specified epochs.

    Args:
        model (SoftmaxRegression or OneHiddenLayerNN): Model instance to train.
        features_train (np.ndarray): Training feature matrix of shape (n_train, d).
        labels_train (np.ndarray): Training label array of shape (n_train,).
        features_val (np.ndarray or None): Validation feature matrix of shape (n_val, d).
        labels_val (np.ndarray or None): Validation label array of shape (n_val,).
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per mini-batch.
        learning_rate (float): Step size for gradient descent updates.
        return_history (bool): If True, returns the training history dictionary.
        use_best_validation (bool): If True, restores weights from the best validation epoch.

    Returns:
        dict[str, list[float]] or None: Training history containing 'train_loss',
            'train_acc', 'val_loss', and 'val_acc' lists if return_history is True,
            otherwise None.
    """
    total_samples = features_train.shape[0]
    best_validation_loss = float('inf')
    best_weights = None

    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(total_samples)
        features_shuffled = features_train[shuffled_indices]
        labels_shuffled = labels_train[shuffled_indices]

        for batch_start in range(0, total_samples, batch_size):
            batch_end = batch_start + batch_size
            features_batch = features_shuffled[batch_start:batch_end]
            labels_batch = labels_shuffled[batch_start:batch_end]

            if hasattr(model, 'weight_matrix_2'):
                gradient_w1, gradient_b1, gradient_w2, gradient_b2 = model.backward(features_batch, labels_batch)
                model.update(gradient_w1, gradient_b1, gradient_w2, gradient_b2, learning_rate)
            else:
                gradient_w, gradient_b = model.backward(features_batch, labels_batch)
                model.update(gradient_w, gradient_b, learning_rate)

        train_loss, train_acc = evaluate(model, features_train, labels_train)
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)

        if features_val is not None and labels_val is not None:
            val_loss, val_acc = evaluate(model, features_val, labels_val)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)

            if use_best_validation and val_loss < best_validation_loss:
                best_validation_loss = val_loss
                if hasattr(model, 'weight_matrix_2'):
                    best_weights = (
                        model.weight_matrix_1.copy(),
                        model.bias_vector_1.copy(),
                        model.weight_matrix_2.copy(),
                        model.bias_vector_2.copy()
                    )
                else:
                    best_weights = (model.weight_matrix.copy(), model.bias_vector.copy())

    if use_best_validation and best_weights is not None:
        if hasattr(model, 'weight_matrix_2'):
            model.weight_matrix_1, model.bias_vector_1, model.weight_matrix_2, model.bias_vector_2 = best_weights
        else:
            model.weight_matrix, model.bias_vector = best_weights

    if return_history:
        return training_history
    return None
