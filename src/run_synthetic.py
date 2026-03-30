"""
Synthetic Dataset Experiment Execution Script.

Loads two-dimensional synthetic clusters and verifies the geometric decision
separability capacity of both linear Softmax Regression and non-linear Neural Networks.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, setup_logger
from src.models.neural import OneHiddenLayerNN
from src.models.linear import SoftmaxRegression
from src.visualization.plotters import plot_decision_boundary
from src.core.trainer import train_model

logger = setup_logger(__name__)


def extract_compressed_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and decompresses a NumPy archive containing train and test splits.

    Args:
        dataset_path (Path): Absolute path to the compressed .npz archive file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features,
            training labels, test features, and test labels.
    """
    try:
        dataset_archive = np.load(dataset_path)
        return dataset_archive['X_train'], dataset_archive['y_train'], dataset_archive['X_test'], dataset_archive['y_test']
    except FileNotFoundError as file_error:
        logger.error(f"Critical System Failure: Compressed Archive not found at sequence path -> {file_error}")
        sys.exit(1)
    except KeyError as archive_key_error:
        logger.error(f"Corrupted Archive Structure: Missing required tensor matrix key -> {archive_key_error}")
        sys.exit(1)


def execute_synthetic_pipeline() -> None:
    """
    Runs all synthetic experiments including Gaussian, Moons, and capacity ablation.

    Returns:
        None
    """
    path_gaussian: Path = DATA_DIR / 'linear_gaussian.npz'
    path_moons: Path = DATA_DIR / 'moons.npz'

    logger.info("Initiating Linear Gaussian Task computations...")
    features_train, labels_train, features_test, labels_test = extract_compressed_dataset(path_gaussian)
    input_dimension_size = features_train.shape[1]
    unique_class_count = len(np.unique(labels_train))

    linear_classifier_gaussian = SoftmaxRegression(input_dimension_size, unique_class_count)
    train_model(linear_classifier_gaussian, features_train, labels_train, epochs=100, use_best_validation=False)
    plot_decision_boundary(
        linear_classifier_gaussian,
        features_test,
        labels_test,
        'Softmax Regression Classification (Gaussian)',
        str(FIGURES_DIR / 'gaussian_softmax.png')
    )

    network_classifier_gaussian = OneHiddenLayerNN(input_dimension_size, 32, unique_class_count)
    train_model(network_classifier_gaussian, features_train, labels_train, epochs=100, use_best_validation=False)
    plot_decision_boundary(
        network_classifier_gaussian,
        features_test,
        labels_test,
        'Neural Network Classification (Gaussian)',
        str(FIGURES_DIR / 'gaussian_nn.png')
    )

    logger.info("Initiating Non-linear Moons Task computations...")
    features_train_moons, labels_train_moons, features_test_moons, labels_test_moons = extract_compressed_dataset(path_moons)

    linear_classifier_moons = SoftmaxRegression(input_dimension_size, unique_class_count)
    train_model(linear_classifier_moons, features_train_moons, labels_train_moons, epochs=200, learning_rate=0.1, use_best_validation=False)
    plot_decision_boundary(
        linear_classifier_moons,
        features_test_moons,
        labels_test_moons,
        'Softmax Regression Classification (Moons)',
        str(FIGURES_DIR / 'moons_softmax.png')
    )

    network_classifier_moons = OneHiddenLayerNN(input_dimension_size, 32, unique_class_count)
    train_model(network_classifier_moons, features_train_moons, labels_train_moons, epochs=200, learning_rate=0.1, use_best_validation=False)
    plot_decision_boundary(
        network_classifier_moons,
        features_test_moons,
        labels_test_moons,
        'Neural Network Classification (Moons)',
        str(FIGURES_DIR / 'moons_nn.png')
    )

    logger.info("Executing Network Capacity Parameter Ablation...")
    hidden_layer_widths = [2, 8, 32]
    for hidden_neurons in hidden_layer_widths:
        ablated_network = OneHiddenLayerNN(input_dimension_size, hidden_neurons, unique_class_count)
        train_model(ablated_network, features_train_moons, labels_train_moons, epochs=200, learning_rate=0.1, use_best_validation=False)
        plot_decision_boundary(
            ablated_network,
            features_test_moons,
            labels_test_moons,
            f'Neural Network Width Expansion w={hidden_neurons} (Moons)',
            str(FIGURES_DIR / f'moons_nn_w{hidden_neurons}.png')
        )

    logger.info("Synthetic evaluation pipeline fully executed.")


if __name__ == "__main__":
    execute_synthetic_pipeline()
