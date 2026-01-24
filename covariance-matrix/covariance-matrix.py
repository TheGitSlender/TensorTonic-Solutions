import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if np.shape(X)[0] < 2 or X.ndim !=2:
        return None
    variable = np.mean(X,axis=0)
    X_centered = X - variable
    return np.dot(X_centered.T, X_centered) / (np.shape(X)[0] - 1)