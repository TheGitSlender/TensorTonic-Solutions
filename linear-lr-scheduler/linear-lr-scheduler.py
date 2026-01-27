def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    if step < warmup_steps:
        return initial_lr * step / max(1, warmup_steps)
    elif step <= total_steps:
        decay_steps = total_steps - warmup_steps
        if decay_steps > 0:
            return final_lr + (initial_lr - final_lr) * (total_steps - step) / decay_steps
        else:
            return final_lr
    else:
        return final_lr