"""
MNIST Digits Benchmark Evaluation.

Executes a robust statistical framework for comparing modeling architectures
against real-world digit image pixels using repeated random seed instantiations.
Also runs the required optimizer study (SGD, Momentum, Adam) on the digits benchmark.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, CROSS_ENTROPY_L2_REGULARIZATION, setup_logger
from src.models.neural import OneHiddenLayerNN
from src.models.linear import SoftmaxRegression
from src.core.metrics import evaluate
from src.core.trainer import train_model

logger = setup_logger(__name__)

BENCHMARK_ITERATION_SEEDS = 5

OPTIMIZER_CONFIGS = [
    {'name': 'SGD',      'optimizer_type': 'sgd',      'learning_rate': 0.05},
    {'name': 'Momentum', 'optimizer_type': 'momentum',  'learning_rate': 0.05},
    {'name': 'Adam',     'optimizer_type': 'adam',      'learning_rate': 0.001},
]


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
    log_lines = []

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
        seed_line = (f"Seed {randomized_seed+1}: Test Accuracy = {terminal_accuracy:.4f}, "
                     f"Test Cross-Entropy = {terminal_entropy:.4f}")
        logger.info(seed_line)
        log_lines.append(seed_line)

    average_accuracy = np.mean(accuracy_measurements)
    deviation_accuracy = np.std(accuracy_measurements, ddof=1)
    confidence_interval_accuracy = 2.776 * (deviation_accuracy / np.sqrt(total_instantiations))

    average_entropy = np.mean(entropy_measurements)
    deviation_entropy = np.std(entropy_measurements, ddof=1)
    confidence_interval_entropy = 2.776 * (deviation_entropy / np.sqrt(total_instantiations))

    summary_line = (f"[{architecture_class.__name__}] Mean Accuracy: {average_accuracy:.4f} "
                    f"+/- {confidence_interval_accuracy:.4f} (95% CI) | "
                    f"Mean Cross-Entropy: {average_entropy:.4f} "
                    f"+/- {confidence_interval_entropy:.4f} (95% CI)")
    logger.info(summary_line)
    log_lines.append(summary_line)

    log_path = RESULTS_DIR / 'digits_log.txt'
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write('\n'.join(log_lines) + '\n\n')


def run_optimizer_study(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    total_pixel_dimensions: int,
    total_classification_categories: int
) -> None:
    """
    Compares SGD, Momentum, and Adam optimizers on the one-hidden-layer NN (width=32).

    All three optimizers use the same train/val split, epoch budget (200), batch size (64),
    L2 regularization, and fixed random seed (42) for a fair comparison.
    Results are saved to figures/digits_optimizer_comparison.png and results/digits_log.txt.

    Args:
        features_train (np.ndarray): Training feature matrix.
        labels_train (np.ndarray): Training labels.
        features_val (np.ndarray): Validation feature matrix.
        labels_val (np.ndarray): Validation labels.
        total_pixel_dimensions (int): Number of input features (64 for digits).
        total_classification_categories (int): Number of output classes (10).

    Returns:
        None
    """
    logger.info("Running Optimizer Study: SGD vs Momentum vs Adam on Digits NN (width=32)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Optimizer Study: SGD vs Momentum vs Adam\n(One-Hidden-Layer NN, width=32, digits benchmark)',
        fontsize=13, fontweight='bold'
    )

    colors = {'SGD': '#e07b39', 'Momentum': '#3a7ebf', 'Adam': '#3da350'}
    final_results = {}

    for config in OPTIMIZER_CONFIGS:
        np.random.seed(42)
        network = OneHiddenLayerNN(
            total_pixel_dimensions, 32,
            total_classification_categories,
            l2_regularization=CROSS_ENTROPY_L2_REGULARIZATION
        )
        network.optimizer_type = config['optimizer_type']

        history = train_model(
            network,
            features_train, labels_train,
            features_val, labels_val,
            epochs=200,
            batch_size=64,
            learning_rate=config['learning_rate'],
            return_history=True,
            use_best_validation=False
        )

        label = config['name']
        color = colors[label]
        axes[0].plot(history['val_loss'],   label=label, color=color, linewidth=2)
        axes[1].plot(history['val_acc'],    label=label, color=color, linewidth=2)

        final_val_loss = history['val_loss'][-1]
        final_val_acc  = history['val_acc'][-1]
        final_results[label] = (final_val_loss, final_val_acc)
        entry = (f"Optimizer={label} | Final Val Accuracy: {final_val_acc:.4f} | "
                 f"Final Val Cross-Entropy: {final_val_loss:.4f}")
        logger.info(entry)

    for ax, ylabel, title in [
        (axes[0], 'Validation Cross-Entropy', 'Validation Loss Dynamics'),
        (axes[1], 'Validation Accuracy',      'Validation Accuracy Dynamics')
    ]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / 'digits_optimizer_comparison.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Optimizer comparison figure saved to {out_path}")

    log_path = RESULTS_DIR / 'digits_log.txt'
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write('[Optimizer Study] SGD vs Momentum vs Adam (width=32, seed=42, 200 epochs)\n')
        for opt_name, (ce, acc) in final_results.items():
            log_file.write(f"  {opt_name}: Val Accuracy={acc:.4f} | Val Cross-Entropy={ce:.4f}\n")
        log_file.write('\n')


def run_failure_case_analysis(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    total_pixel_dimensions: int,
    total_classification_categories: int
) -> None:
    """
    Reproduces the 'Optimization Failure' case using an extremely low learning rate.

    This demonstrates that even a high-capacity model (width=32) fails to learn
    if the optimization hyperparameter (lr=0.005) is poorly chosen, resulting in
    stagnant loss and uniform-random predictions.
    """
    logger.info("Running Failure Case Analysis: Low Learning Rate (lr=0.005) Underfitting...")

    np.random.seed(42)
    failed_network = OneHiddenLayerNN(
        total_pixel_dimensions, 32,
        total_classification_categories,
        l2_regularization=CROSS_ENTROPY_L2_REGULARIZATION
    )

    history = train_model(
        failed_network,
        features_train, labels_train,
        features_val, labels_val,
        epochs=100,
        batch_size=64,
        learning_rate=0.005,
        return_history=True,
        use_best_validation=False
    )

    final_val_loss = history['val_loss'][-1]
    logger.warning(
        f"FAILURE CASE CONFIRMED: With lr=0.005, final validation entropy is {final_val_loss:.4f} "
        "(Compare to ~0.09 with Adam). The model failed to converge."
    )


def execute_digits_benchmark() -> None:
    """
    Coordinates the full digits benchmark: repeated-seed evaluation + optimizer study.

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

    run_optimizer_study(
        features_train, labels_train,
        features_val, labels_val,
        total_pixel_dimensions,
        total_classification_categories
    )

    run_failure_case_analysis(
        features_train, labels_train,
        features_val, labels_val,
        total_pixel_dimensions,
        total_classification_categories
    )


if __name__ == "__main__":
    execute_digits_benchmark()
