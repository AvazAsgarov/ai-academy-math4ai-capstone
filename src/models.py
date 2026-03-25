import numpy as np

def softmax(scores):
    """
    Computes numerically stable softmax probabilities.
    scores: (n, k) array of logits.
    """
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, input_dim, num_classes, l2_reg=1e-4):
        # Xavier-like initialization for linear models
        self.W = np.random.randn(num_classes, input_dim) * np.sqrt(1.0 / input_dim)
        self.b = np.zeros(num_classes)
        self.l2_reg = l2_reg

    def forward(self, X):
        """
        X: (n, d)
        Returns: S (n, k)
        """
        return X.dot(self.W.T) + self.b

    def predict_probs(self, X):
        S = self.forward(X)
        return softmax(S)

    def predict(self, X):
        P = self.predict_probs(X)
        return np.argmax(P, axis=1)

    def compute_loss(self, X, y):
        n = X.shape[0]
        P = self.predict_probs(X)
        # Avoid log(0) with 1e-15
        core_loss = -np.sum(np.log(P[np.arange(n), y] + 1e-15)) / n
        reg_loss = 0.5 * self.l2_reg * np.sum(self.W ** 2)
        return core_loss + reg_loss

    def backward(self, X, y):
        n = X.shape[0]
        P = self.predict_probs(X)
        
        # dS = (P - Y) / n
        dS = P.copy()
        dS[np.arange(n), y] -= 1
        dS /= n
        
        dW = dS.T.dot(X) + self.l2_reg * self.W
        db = np.sum(dS, axis=0)
        
        return dW, db

    def update(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db