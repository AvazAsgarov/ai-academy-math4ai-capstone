import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, X, y):
    loss = model.compute_loss(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    return loss, acc

def train_model(model, X_train, y_train, X_val=None, y_val=None, 
                epochs=200, batch_size=64, lr=0.05, return_history=False, use_best_val=True):
    n_samples = X_train.shape[0]
    best_val_loss = float('inf')
    best_weights = None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            if hasattr(model, 'W2'):
                dW1, db1, dW2, db2 = model.backward(X_batch, y_batch)
                model.update(dW1, db1, dW2, db2, lr)
            else:
                dW, db = model.backward(X_batch, y_batch)
                model.update(dW, db, lr)
                
        train_loss, train_acc = evaluate(model, X_train, y_train)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        if X_val is not None and y_val is not None:
            val_loss, val_acc = evaluate(model, X_val, y_val)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if use_best_val and val_loss < best_val_loss:
                best_val_loss = val_loss
                if hasattr(model, 'W2'):
                    best_weights = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())
                else:
                    best_weights = (model.W.copy(), model.b.copy())
                    
    if use_best_val and best_weights is not None:
        if hasattr(model, 'W2'):
            model.W1, model.b1, model.W2, model.b2 = best_weights
        else:
            model.W, model.b = best_weights
            
    if return_history:
        return history

def plot_decision_boundary(model, X, y, title, filepath):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
                         
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k', cmap=plt.cm.Spectral)
    plt.title(title)
    plt.savefig(filepath)
    plt.close()
