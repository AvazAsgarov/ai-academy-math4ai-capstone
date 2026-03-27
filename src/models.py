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
        # Xavier-like initialization
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
        # Avoiding log(0) with 1e-15
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
class OneHiddenLayerNN:
    def __init__(self, input_dim, hidden_dim, num_classes, l2_reg=1e-4):

        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        
        limit2 = np.sqrt(6.0 / (hidden_dim + num_classes))
        self.W2 = np.random.uniform(-limit2, limit2, (num_classes, hidden_dim))
        self.b2 = np.zeros(num_classes)
        
        self.l2_reg = l2_reg

    def forward(self, X):
        Z1 = X.dot(self.W1.T) + self.b1
        H = np.tanh(Z1)
        S = H.dot(self.W2.T) + self.b2
        return Z1, H, S

    def predict_probs(self, X):
        _, _, S = self.forward(X)
        return softmax(S)

    def predict(self, X):
        P = self.predict_probs(X)
        return np.argmax(P, axis=1)

    def compute_loss(self, X, y):
        n = X.shape[0]
        P = self.predict_probs(X)
        core_loss = -np.sum(np.log(P[np.arange(n), y] + 1e-15)) / n
        reg_loss = 0.5 * self.l2_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return core_loss + reg_loss

    def backward(self, X, y):
        n = X.shape[0]
        Z1, H, S = self.forward(X)
        P = softmax(S)
        
        # Output layer gradients
        dS = P.copy()
        dS[np.arange(n), y] -= 1
        dS /= n
        
        dW2 = dS.T.dot(H) + self.l2_reg * self.W2
        db2 = np.sum(dS, axis=0)
        
        # Hidden layer gradients
        dZ1 = dS.dot(self.W2) * (1 - H**2)
        dW1 = dZ1.T.dot(X) + self.l2_reg * self.W1
        db1 = np.sum(dZ1, axis=0)
        
        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
