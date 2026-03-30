"""
MNIST Digits Benchmark Evaluation.

Executes a robust statistical framework for comparing modeling architectures
against real-world digit image pixels using repeated random seed instantiations.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, CROSS_ENTROPY_L2_REGULARIZATION, setup_logger
from src.models.neural import OneHiddenLayerNN
from src.models.linear import SoftmaxRegression
from src.core.metrics import evaluate
from src.core.trainer import train_model

logger = setup_logger(__name__)

LEARNING_RATE_ABLATION_SWEEPS = [0.005, 0.05, 0.2]
BENCHMARK_ITERATION_SEEDS = 5


def load_digit_arrays() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Loads and splits the digits dataset into train, validation, and test subsets.

    Returns:
        tuple: Three tuples of (features, labels) for train, validation, and test splits,
            where features are np.ndarray of shape (n_split, 64) and labels are np.ndarray
            of shape (n_split,).
    """
    dataset_filepath = DATA_DIR / 'digits_data.npz'
    index_split_filepath = DATA_DIR / 'digits_split_indices.npz'

    try:
        extracted_data = np.load(dataset_filepath)
        extracted_splits = np.load(index_split_filepath)

        pixel_features = extracted_data['X']
        class_labels = extracted_data['y']

        indices_train = extracted_splits['train_idx']
        indices_validate = extracted_splits['val_idx']
        indices_test = extracted_splits['test_idx']

        return (
            (pixel_features[indices_train], class_labels[indices_train]),
            (pixel_features[indices_validate], class_labels[indices_validate]),
            (pixel_features[indices_test], class_labels[indices_test])
        )
    except FileNotFoundError as err:
        logger.error(f"Fatal System Error: Dataset archive matrix missing -> {err}")
        sys.exit(1)
    except KeyError as key_err:
        logger.error(f"Archive Integrity Compromised: Missing internal array slice pointer -> {key_err}")
        sys.exit(1)


def evaluate_statistical_seeds(
    architecture_class,
    architecture_parameters: dict,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    total_instantiations: int = BENCHMARK_ITERATION_SEEDS,
    descent_rate: float = 0.05
) -> None:
    """
    Evaluates a model architecture across multiple random seeds and reports aggregated statistics.

    Args:
        architecture_class (type): Model class to instantiate (SoftmaxRegression or OneHiddenLayerNN).
        architecture_parameters (dict): Keyword arguments passed to the model constructor.
        features_train (np.ndarray): Training feature matrix of shape (n_train, d).
        labels_train (np.ndarray): Training label array of shape (n_train,).
        features_val (np.ndarray): Validation feature matrix of shape (n_val, d).
        labels_val (np.ndarray): Validation label array of shape (n_val,).
        features_test (np.ndarray): Test feature matrix of shape (n_test, d).
        labels_test (np.ndarray): Test label array of shape (n_test,).
        total_instantiations (int): Number of independent random seeds to evaluate.
        descent_rate (float): Learning rate for gradient descent optimization.

    Returns:
        None
    """
    logger.info(f"Executing {total_instantiations} independent evaluations for {architecture_class.__name__}...")
    accuracy_measurements = []
    entropy_measurements = []

    for randomized_seed in range(total_instantiations):
        np.random.seed(randomized_seed)
        classifier_instance = architecture_class(**architecture_parameters)
        train_model(
            classifier_instance,
            features_train,
            labels_train,
            features_val,
            labels_val,
            epochs=200,
            batch_size=64,
            learning_rate=descent_rate,
            use_best_validation=True
        )

        terminal_entropy, terminal_accuracy = evaluate(classifier_instance, features_test, labels_test)
        accuracy_measurements.append(terminal_accuracy)
        entropy_measurements.append(terminal_entropy)
        logger.info(f"Algorithm Instance {randomized_seed+1}: Holdout Accuracy = {terminal_accuracy:.4f}, Holdout Entropy = {terminal_entropy:.4f}")

    average_accuracy = np.mean(accuracy_measurements)
    deviation_accuracy = np.std(accuracy_measurements, ddof=1)
    confidence_interval_accuracy = 2.776 * (deviation_accuracy / np.sqrt(total_instantiations))

    average_entropy = np.mean(entropy_measurements)
    deviation_entropy = np.std(entropy_measurements, ddof=1)
    confidence_interval_entropy = 2.776 * (deviation_entropy / np.sqrt(total_instantiations))

    logger.info(f"Aggregated Performance Metrics [{architecture_class.__name__}]: "
                f"Mean Accuracy: {average_accuracy:.4f} \u00B1 {confidence_interval_accuracy:.4f} | "
                f"Mean Entropy: {average_entropy:.4f} \u00B1 {confidence_interval_entropy:.4f}")


def execute_digits_benchmark() -> None:
    """
    Coordinates the full digits benchmark including seed evaluation and learning rate ablation.

    Returns:
        None
    """
    (features_train, labels_train), (features_val, labels_val), (features_test, labels_test) = load_digit_arrays()
    total_pixel_dimensions = features_train.shape[1]
    total_classification_categories = 10

    evaluate_statistical_seeds(
        SoftmaxRegression,
        {'input_dimensions': total_pixel_dimensions, 'total_classes': total_classification_categories, 'l2_regularization': CROSS_ENTROPY_L2_REGULARIZATION},
        features_train, labels_train, features_val, labels_val, features_test, labels_test,
        descent_rate=0.1
    )

    evaluate_statistical_seeds(
        OneHiddenLayerNN,
        {'input_dimensions': total_pixel_dimensions, 'hidden_dimensions': 32, 'total_classes': total_classification_categories, 'l2_regularization': CROSS_ENTROPY_L2_REGULARIZATION},
        features_train, labels_train, features_val, labels_val, features_test, labels_test,
        descent_rate=0.1
    )

    logger.info("Executing Descent Rate Dynamical Ablation mapping Phase...")

    try:
        plt.figure(figsize=(10, 6))
        for specified_rate in LEARNING_RATE_ABLATION_SWEEPS:
            np.random.seed(42)
            ablated_network = OneHiddenLayerNN(total_pixel_dimensions, 32, total_classification_categories)
            training_history_log = train_model(
                ablated_network,
                features_train,
                labels_train,
                features_val,
                labels_val,
                epochs=100,
                batch_size=64,
                learning_rate=specified_rate,
                return_history=True,
                use_best_validation=False
            )
            plt.plot(training_history_log['val_loss'], label=f'Descent Speed = {specified_rate}')
            logger.info(f"Speed Rate {specified_rate}: Ultimate Validation Entropy Metric: {training_history_log['val_loss'][-1]:.4f}")

        plt.title('Validation Cross-Entropy Dynamics Distinguishing Parameter Updates')
        plt.xlabel('Iterative Epoch Configuration')
        plt.ylabel('Recorded Validation Entropy Measure')
        plt.legend()
        plt.savefig(str(FIGURES_DIR / 'lr_ablation.png'))
        plt.close()
    except Exception as graphical_err:
        logger.error(f"Failed to compile graphic outputs. Matplotlib runtime crash -> {graphical_err}")


if __name__ == "__main__":
    execute_digits_benchmark()
