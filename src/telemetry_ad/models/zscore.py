import numpy as np


def fit_robust_baseline(x: np.ndarray) -> dict:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return {"median": med, "mad": max(mad, 1e-8)}


def score_robust_z(x: np.ndarray, params: dict) -> np.ndarray:
    return 0.6745 * np.abs((x - params["median"]) / params["mad"])
