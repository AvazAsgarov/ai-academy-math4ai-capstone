"""
Decision Boundary Visualization.

Provides tools for rendering and saving 2D classification decision boundaries.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    plot_title: str,
    save_filepath: str
) -> None:
    """
    Renders the model decision boundaries over a 2D feature space and saves the plot.

    Args:
        model (SoftmaxRegression or OneHiddenLayerNN): Trained classifier instance.
        features (np.ndarray): Input feature matrix of shape (n, 2).
        labels (np.ndarray): Ground-truth class label array of shape (n,).
        plot_title (str): Title displayed above the generated figure.
        save_filepath (str): Absolute or relative path where the PNG image is saved.
    """
    x_axis_min = features[:, 0].min() - 0.5
    x_axis_max = features[:, 0].max() + 0.5
    y_axis_min = features[:, 1].min() - 0.5
    y_axis_max = features[:, 1].max() + 0.5

    mesh_grid_x, mesh_grid_y = np.meshgrid(
        np.arange(x_axis_min, x_axis_max, 0.02),
        np.arange(y_axis_min, y_axis_max, 0.02)
    )

    grid_coordinates = np.c_[mesh_grid_x.ravel(), mesh_grid_y.ravel()]
    predictions = model.predict(grid_coordinates)
    prediction_surface = predictions.reshape(mesh_grid_x.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(mesh_grid_x, mesh_grid_y, prediction_surface, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(features[:, 0], features[:, 1], c=labels, s=40, edgecolors='k', cmap=plt.cm.Spectral)

    plt.title(plot_title)
    plt.savefig(save_filepath)
    plt.close()
