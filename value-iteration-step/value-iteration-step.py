def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    result = []
    for state in range(len(transitions)):
        q = float("-inf")
        for action in range(len(transitions[state])):
            temp_q = rewards[state][action]
            for next_state in range(len(transitions[state][action])):
                temp_q += gamma * transitions[state][action][next_state] * values[next_state]
            if temp_q > q:
                q = temp_q
        result.append(q)
    return result