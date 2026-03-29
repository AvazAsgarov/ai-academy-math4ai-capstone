import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import SoftmaxRegression
from src.utils import train_model, evaluate

def pca_svd(X, n_components):
    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    return U, S, Vt, mean

def run_track_a():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'digits_data.npz')
    split_path = os.path.join(base_dir, 'data', 'digits_split_indices.npz')
    
    data = np.load(data_path)
    splits = np.load(split_path)
    
    X = data['X']
    y = data['y']
    
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 1. PCA/SVD
    _, S, Vt, mean = pca_svd(X_train, X_train.shape[1])
    
    # Scree Plot
    explained_variance = (S ** 2) / (X_train.shape[0] - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(explained_variance_ratio), marker='o')
    plt.title('Scree Plot (Cumulative Explained Variance)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'scree_plot.png'))
    plt.close()
    
    # 2D PCA Visualization
    X_train_centered = X_train - mean
    X_train_pca2 = X_train_centered.dot(Vt[:2].T)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_pca2[:, 0], X_train_pca2[:, 1], c=y_train, cmap='tab10', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('2D PCA Visualization of Digits Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(os.path.join(figures_dir, 'pca_2d.png'))
    plt.close()
    
    # Softmax on PCA dimensions m = {10, 20, 40}
    print("\nTrack A: Softmax Comparison at fixed PCA dimensions m in {10, 20, 40}")
    m_values = [10, 20, 40]
    for m in m_values:
        V_m = Vt[:m].T
        X_train_m = (X_train - mean).dot(V_m)
        X_val_m = (X_val - mean).dot(V_m)
        X_test_m = (X_test - mean).dot(V_m)
        
        sm = SoftmaxRegression(m, 10, l2_reg=1e-4)
        np.random.seed(42)
        train_model(sm, X_train_m, y_train, X_val_m, y_val, epochs=200, batch_size=64, lr=0.1, use_best_val=True)
        ce, acc = evaluate(sm, X_test_m, y_test)
        print(f"PCA m={m} | Test Acc: {acc:.4f} | Test CE: {ce:.4f}")

if __name__ == '__main__':
    run_track_a()
