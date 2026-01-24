import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    rewards, V = np.asarray(rewards), np.asarray(V)
    discount_sum = np.zeros(len(rewards))
    length = len(states)
    running_add = 0
    for t in reversed(range(length)):
        running_add = rewards[t] + gamma*running_add
        discount_sum[t] = running_add
    return discount_sum - V
    return (np.asarray(result))