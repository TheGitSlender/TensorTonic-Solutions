import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    x, h_prev = np.asarray(x), np.asarray(h_prev)

    if x.ndim == 1:
        x, _ = _as2d(x, x.shape[0])
    if h_prev.ndim == 1:
        h_prev, _ = _as2d(h_prev, h_prev.shape[0])

    zt = _sigmoid(np.dot(x,params["Wz"]) + np.dot(h_prev,params["Uz"]) + params["bz"])
    rt = _sigmoid(np.dot(x,params["Wr"]) + np.dot(h_prev,params["Ur"]) + params["br"])
    h = np.tanh(np.dot(x,params["Wh"]) + np.dot((rt*h_prev),params["Uh"]) + params["bh"])
    final_h = ((1-zt)*h_prev) + (zt*h)
    if final_h.shape[0] == 1:
        return final_h.squeeze()
    return final_h
