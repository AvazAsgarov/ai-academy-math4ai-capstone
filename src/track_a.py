"""
Advanced Analytical PCA Track.

Conducts Singular Value Decomposition to orthogonally project digit image
tensors into low-dimensional representations while preserving variance.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, CROSS_ENTROPY_L2_REGULARIZATION, GLOBAL_RANDOM_SEED, setup_logger
from src.models.linear import SoftmaxRegression
from src.core.metrics import evaluate
from src.core.trainer import train_model

logger = setup_logger(__name__)

PCA_MANIFOLD_DIMENSIONS = [10, 20, 40]


def execute_principal_component_analysis(features: np.ndarray, retained_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs PCA via SVD on mean-centered features.

    Args:
        features (np.ndarray): Input feature matrix of shape (n, d).
        retained_components (int): Maximum number of principal components to retain.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Left singular vectors (n, d),
            singular values (min(n,d),), right singular vectors transposed (d, d),
            and per-feature mean vector (d,).
    """
    dimension_means = np.mean(features, axis=0)
    centered_features = features - dimension_means

    left_singular, singular_values, right_singular_transposed = np.linalg.svd(centered_features, full_matrices=False)

    return left_singular, singular_values, right_singular_transposed, dimension_means


def launch_advanced_pca_investigations() -> None:
    """
    Runs the full Track A pipeline including scree plot, 2D visualization, and PCA ablation.

    Returns:
        None
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

        features_train, labels_train = pixel_features[indices_train], class_labels[indices_train]
        features_val, labels_val = pixel_features[indices_validate], class_labels[indices_validate]
        features_test, labels_test = pixel_features[indices_test], class_labels[indices_test]
    except (FileNotFoundError, KeyError) as err:
        logger.error(f"Critical data ingestion failure. Exiting PCA Track -> {err}")
        sys.exit(1)

    _, singular_spectrum, principal_axes, feature_means = execute_principal_component_analysis(features_train, features_train.shape[1])

    variance_magnitudes = (singular_spectrum ** 2) / (features_train.shape[0] - 1)
    normalized_variance_ratios = variance_magnitudes / np.sum(variance_magnitudes)

    try:
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(normalized_variance_ratios), marker='o')
        plt.title('Scree Plot Detailing Cumulative Explained Phenomenological Variance')
        plt.xlabel('Quantity of Extracted Components')
        plt.ylabel('Accumulated Variance Fraction')
        plt.grid(True)
        plt.savefig(str(FIGURES_DIR / 'scree_plot.png'))
        plt.close()

        centered_train = features_train - feature_means
        planar_projections = centered_train.dot(principal_axes[:2].T)

        plt.figure(figsize=(8, 6))
        scatter_plot = plt.scatter(planar_projections[:, 0], planar_projections[:, 1], c=labels_train, cmap='tab10', alpha=0.6, s=15)
        plt.colorbar(scatter_plot, label='Digit Categorical Class')
        plt.title('Orthogonal 2D Visualization Mapping Distinct Digit Clouds')
        plt.xlabel('Dominant Principal Component')
        plt.ylabel('Secondary Principal Component')
        plt.savefig(str(FIGURES_DIR / 'pca_2d.png'))
        plt.close()
    except Exception as e:
        logger.error(f"Visualization render engine failed -> {e}")

    logger.info(f"Executing Track A Assessments: Benchmarking Generalized Softmax Across Configured Reduced Capacities m={PCA_MANIFOLD_DIMENSIONS}")
    total_categories = 10

    for dimension_slice in PCA_MANIFOLD_DIMENSIONS:
        sliced_orthogonal_basis = principal_axes[:dimension_slice].T
        compact_train = (features_train - feature_means).dot(sliced_orthogonal_basis)
        compact_validate = (features_val - feature_means).dot(sliced_orthogonal_basis)
        compact_test = (features_test - feature_means).dot(sliced_orthogonal_basis)

        linear_classifier = SoftmaxRegression(dimension_slice, total_categories, l2_regularization=CROSS_ENTROPY_L2_REGULARIZATION)
        np.random.seed(GLOBAL_RANDOM_SEED)
        train_model(linear_classifier, compact_train, labels_train, compact_validate, labels_val, epochs=200, batch_size=64, learning_rate=0.1, use_best_validation=True)
        terminal_entropy, terminal_accuracy = evaluate(linear_classifier, compact_test, labels_test)
        logger.info(f"Subspace Configuration m={dimension_slice} | Validation Accuracy: {terminal_accuracy:.4f} | Measured Entropy: {terminal_entropy:.4f}")


if __name__ == '__main__':
    launch_advanced_pca_investigations()
