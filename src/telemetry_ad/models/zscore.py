import numpy as np


def fit_robust_baseline(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=float)
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0)
    mad = np.maximum(mad, 1e-8)
    return {"median": med, "mad": mad}


def score_robust_z(x: np.ndarray, params: dict) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    z = 0.6745 * np.abs((arr - params["median"]) / params["mad"])
    if z.ndim == 1:
        return z
    return z.max(axis=1)
