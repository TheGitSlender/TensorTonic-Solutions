import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t, h_prev, Wx, Wh, b = np.asarray(x_t), np.asarray(h_prev), np.asarray(Wx), np.asarray(Wh), np.asarray(b)
    return np.tanh(np.dot(x_t,Wx) + np.dot(h_prev,Wh) + b)
