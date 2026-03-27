import numpy as np
import os
import sys

# Ensure src is in pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import SoftmaxRegression, OneHiddenLayerNN
from src.utils import train_model, evaluate, plot_decision_boundary

def load_data(path):
    data = np.load(path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def run_synthetic_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gaussian_path = os.path.join(base_dir, 'data', 'linear_gaussian.npz')
    moons_path = os.path.join(base_dir, 'data', 'moons.npz')
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Running Linear Gaussian Task...")
    X_train, y_train, X_test, y_test = load_data(gaussian_path)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Softmax
    sm = SoftmaxRegression(input_dim, num_classes)
    train_model(sm, X_train, y_train, epochs=100, use_best_val=False)
    plot_decision_boundary(sm, X_test, y_test, 'Softmax Regression (Gaussian)', 
                           os.path.join(figures_dir, 'gaussian_softmax.png'))
                           
    # NN
    nn = OneHiddenLayerNN(input_dim, 32, num_classes)
    train_model(nn, X_train, y_train, epochs=100, use_best_val=False)
    plot_decision_boundary(nn, X_test, y_test, 'Neural Network (Gaussian)', 
                           os.path.join(figures_dir, 'gaussian_nn.png'))
                           
    print("Running Moons Task...")
    X_train, y_train, X_test, y_test = load_data(moons_path)
    
    # Softmax
    sm_moons = SoftmaxRegression(input_dim, num_classes)
    train_model(sm_moons, X_train, y_train, epochs=200, lr=0.1, use_best_val=False)
    plot_decision_boundary(sm_moons, X_test, y_test, 'Softmax Regression (Moons)', 
                           os.path.join(figures_dir, 'moons_softmax.png'))
                           
    # NN
    nn_moons = OneHiddenLayerNN(input_dim, 32, num_classes)
    train_model(nn_moons, X_train, y_train, epochs=200, lr=0.1, use_best_val=False)
    plot_decision_boundary(nn_moons, X_test, y_test, 'Neural Network (Moons)', 
                           os.path.join(figures_dir, 'moons_nn.png'))
                           
    # Capacity Ablation on Moons (2, 8, 32)
    print("Running Capacity Ablation on Moons...")
    for width in [2, 8, 32]:
        nn_abl = OneHiddenLayerNN(input_dim, width, num_classes)
        train_model(nn_abl, X_train, y_train, epochs=200, lr=0.1, use_best_val=False)
        plot_decision_boundary(nn_abl, X_test, y_test, f'Neural Network w={width} (Moons)', 
                               os.path.join(figures_dir, f'moons_nn_w{width}.png'))
                               
    print("Synthetic experiments completed.")

if __name__ == "__main__":
    run_synthetic_experiments()
