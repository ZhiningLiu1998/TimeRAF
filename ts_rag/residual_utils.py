import numpy as np


def future_residual(y_true, y_pred):
    return y_true - y_pred


def moving_average_residual(x, kernel_size=5):
    if kernel_size <= 1:
        return x.copy()
    pad = kernel_size // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.zeros_like(x)
    for t in range(x.shape[0]):
        smoothed[t] = padded[t : t + kernel_size].mean(axis=0)
    return x - smoothed


def l2_event_score(residual_seq):
    return np.linalg.norm(residual_seq, axis=-1)


def percentile_threshold(values, percentile):
    return float(np.percentile(values, percentile))


def standardize_patch(patch, eps=1e-6):
    mean = patch.mean(axis=0, keepdims=True)
    std = patch.std(axis=0, keepdims=True)
    return (patch - mean) / (std + eps)
