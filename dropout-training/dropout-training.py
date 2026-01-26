import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x)
    if rng == None:
        dropout_vector = np.random.random(x.shape)
    else:
        dropout_vector = rng.random(x.shape)
    dropout_vector = np.where(dropout_vector<1-p,1/(1-p),0)
    return (np.multiply(x,dropout_vector),dropout_vector)