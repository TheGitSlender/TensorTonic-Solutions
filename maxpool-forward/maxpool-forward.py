import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    X = np.array(X)
    height, width = X.shape
    
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1
    
    output = np.zeros((out_height, out_width))
    
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            w_start = j * stride
            window = X[h_start:h_start + pool_size, w_start:w_start + pool_size]
            output[i, j] = np.max(window)
    
    return output.tolist()