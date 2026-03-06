import numpy as np


def sliding_windows(x: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    if len(x) < window_size:
        return np.empty((0, window_size) + x.shape[1:])
    windows = []
    for i in range(0, len(x) - window_size + 1, stride):
        windows.append(x[i : i + window_size])
    return np.asarray(windows)
