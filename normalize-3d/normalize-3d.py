import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v)
    if v.shape == (3,):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v/norm
    else:
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return v/norm