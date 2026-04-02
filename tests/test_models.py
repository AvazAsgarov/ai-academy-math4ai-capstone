"""
Basic correctness tests for models.py and utils.py.

Run from the root of the project:
    python tests/test_models.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.math_ops.utilities import compute_stable_softmax
from src.models.linear import SoftmaxRegression
from src.models.neural import OneHiddenLayerNN


def test_softmax_values():
    """Section 3.6 worked example: s = [1.2, 0.2, -0.4] -> p ~ [0.64, 0.23, 0.13]"""
    S = np.array([[1.2, 0.2, -0.4]])
    P = compute_stable_softmax(S)
    expected = np.array([[0.64, 0.23, 0.13]])
    assert np.allclose(P, expected, atol=1e-2), f"Expected ~{expected}, got {P}"
    print("PASS  test_softmax_values")


def test_softmax_sums_to_one():
    """Each row of softmax output must sum to exactly 1."""
    rng = np.random.default_rng(0)
    S = rng.standard_normal((50, 10))
    P = compute_stable_softmax(S)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10), "Rows do not sum to 1"
    print("PASS  test_softmax_sums_to_one")


def test_softmax_numerical_stability():
    """Softmax must not produce NaN or Inf on large inputs."""
    S = np.array([[1000.0, 1000.0, 1000.0],
                  [-1000.0, -1000.0, -1000.0]])
    P = compute_stable_softmax(S)
    assert not np.any(np.isnan(P)), "NaN detected in softmax output"
    assert not np.any(np.isinf(P)), "Inf detected in softmax output"
    print("PASS  test_softmax_numerical_stability")


def test_cross_entropy_value():
    """Section 3.6: true class=0, p~0.64 -> loss ~ 0.45"""
    model = SoftmaxRegression(input_dimensions=2, total_classes=3, l2_regularization=0.0)
    # Force logits to yield specific P
    model.weight_matrix = np.zeros((3, 2))
    model.bias_vector = np.array([1.2, 0.2, -0.4])
    X = np.zeros((1, 2))  # output should exactly equal the bias vector
    y = np.array([0])
    loss = model.compute_loss(X, y)
    assert abs(loss - 0.45) < 0.01, f"Expected ~0.45, got {loss:.4f}"
    print("PASS  test_cross_entropy_value")


def test_forward_shapes():
    """Forward pass must return correct shapes."""
    model = SoftmaxRegression(input_dimensions=4, total_classes=3, l2_regularization=0.0)
    X = np.random.default_rng(0).standard_normal((10, 4))
    S = model.forward(X)
    P = model.predict_probs(X)
    assert S.shape == (10, 3), f"S shape: expected (10,3), got {S.shape}"
    assert P.shape == (10, 3), f"P shape: expected (10,3), got {P.shape}"
    print("PASS  test_forward_shapes")


def test_forward_probs_sum_to_one():
    """Predicted probabilities must sum to 1 for every example."""
    model = SoftmaxRegression(input_dimensions=4, total_classes=3, l2_regularization=0.0)
    X = np.random.default_rng(1).standard_normal((20, 4))
    P = model.predict_probs(X)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10), "P rows do not sum to 1"
    print("PASS  test_forward_probs_sum_to_one")


def test_backward_shapes():
    """Gradient shapes must match parameter shapes."""
    d, k, n = 4, 3, 10
    model = SoftmaxRegression(input_dimensions=d, total_classes=k, l2_regularization=0.0)
    X = np.random.default_rng(2).standard_normal((n, d))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    grad_W, grad_b = model.backward(X, y)
    assert grad_W.shape == (k, d), f"dW shape: expected ({k},{d}), got {grad_W.shape}"
    assert grad_b.shape == (k,),   f"db shape: expected ({k},), got {grad_b.shape}"
    print("PASS  test_backward_shapes")


def test_loss_decreases():
    """Loss must decrease after a few gradient steps on tiny data."""
    d, k, n = 2, 2, 8
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, d))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    model = SoftmaxRegression(input_dimensions=d, total_classes=k, l2_regularization=0.0)
    lr = 0.1

    loss_before = model.compute_loss(X, y)

    for _ in range(20):
        grad_W, grad_b = model.backward(X, y)
        model.update(grad_W, grad_b, learning_rate=lr)

    loss_after = model.compute_loss(X, y)

    assert loss_after < loss_before, (
        f"Loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"
    )
    print(f"PASS  test_loss_decreases  ({loss_before:.4f} -> {loss_after:.4f})")


def test_nn_backward_shapes():
    """Neural Network Gradient shapes must match parameter shapes."""
    d, h, k, n = 4, 8, 3, 10
    model = OneHiddenLayerNN(input_dimensions=d, hidden_dimensions=h, total_classes=k, l2_regularization=0.0)
    X = np.random.default_rng(4).standard_normal((n, d))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    grad_W1, grad_b1, grad_W2, grad_b2 = model.backward(X, y)
    assert grad_W1.shape == (h, d), f"dW1 shape: expected ({h},{d}), got {grad_W1.shape}"
    assert grad_b1.shape == (h,),   f"db1 shape: expected ({h},), got {grad_b1.shape}"
    assert grad_W2.shape == (k, h), f"dW2 shape: expected ({k},{h}), got {grad_W2.shape}"
    assert grad_b2.shape == (k,),   f"db2 shape: expected ({k},), got {grad_b2.shape}"
    print("PASS  test_nn_backward_shapes")


def test_nn_loss_decreases():
    """Neural Network loss must decrease after a few gradient steps on tiny data."""
    d, h, k, n = 2, 8, 2, 8
    rng = np.random.default_rng(5)
    X = rng.standard_normal((n, d))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    model = OneHiddenLayerNN(input_dimensions=d, hidden_dimensions=h, total_classes=k, l2_regularization=0.0)
    lr = 0.1

    loss_before = model.compute_loss(X, y)

    for _ in range(20):
        gW1, gb1, gW2, gb2 = model.backward(X, y)
        model.update(gW1, gb1, gW2, gb2, learning_rate=lr)

    loss_after = model.compute_loss(X, y)

    assert loss_after < loss_before, (
        f"NN Loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"
    )
    print(f"PASS  test_nn_loss_decreases ({loss_before:.4f} -> {loss_after:.4f})")


def test_nn_overfitting():
    """Neural Network must be able to overfit a tiny batch to near-zero loss."""
    d, h, k, n = 4, 32, 2, 4
    rng = np.random.default_rng(6)
    X = rng.standard_normal((n, d))
    y = np.array([0, 1, 0, 1])

    model = OneHiddenLayerNN(input_dimensions=d, hidden_dimensions=h, total_classes=k, l2_regularization=0.0)
    lr = 0.5  # larger lr for faster overfitting

    for _ in range(100):
        gW1, gb1, gW2, gb2 = model.backward(X, y)
        model.update(gW1, gb1, gW2, gb2, learning_rate=lr)

    final_loss = model.compute_loss(X, y)
    assert final_loss < 0.05, f"NN failed to overfit tiny batch: final_loss={final_loss:.4f}"
    print(f"PASS  test_nn_overfitting (Final Loss: {final_loss:.4f})")


if __name__ == "__main__":
    print("\n--- math_utils sanity tests ---")
    test_softmax_values()
    test_softmax_sums_to_one()
    test_softmax_numerical_stability()
    test_cross_entropy_value()
    
    print("\n--- SoftmaxRegression tests ---")
    test_forward_shapes()
    test_forward_probs_sum_to_one()
    test_backward_shapes()
    test_loss_decreases()

    print("\n--- OneHiddenLayerNN tests ---")
    test_nn_backward_shapes()
    test_nn_loss_decreases()
    test_nn_overfitting()
    
    print("\nAll 11 tests passed successfully!")
