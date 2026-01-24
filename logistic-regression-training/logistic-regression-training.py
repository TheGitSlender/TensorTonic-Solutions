import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X, y = np.asarray(X), np.asarray(y)
    n_samples, n_features = X.shape
    w, b = np.zeros(n_features), 0.0
    for _ in range(steps):
        p = _sigmoid(np.dot(X,w)+b)
        grad_w = np.dot(X.T, p - y)/n_samples
        grad_b = np.sum(p-y)/n_samples
        w -= lr * grad_w
        b -= lr * grad_b
    return (w,b)