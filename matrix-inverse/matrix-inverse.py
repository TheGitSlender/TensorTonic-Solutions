import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A)
    if A.ndim != 2 or (A.shape[0] != A.shape[1]) or np.abs(np.linalg.det(A)) < 1e-10:
        return None
    return np.linalg.inv(A)
