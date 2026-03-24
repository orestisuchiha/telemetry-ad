import numpy as np
import pandas as pd


def fft_window_energy(values: np.ndarray, include_dc: bool = False) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0

    spectrum_input = arr if include_dc else arr - np.mean(arr)
    spectrum = np.fft.rfft(spectrum_input)
    if not include_dc and spectrum.size > 0:
        spectrum = spectrum[1:]
    if spectrum.size == 0:
        return 0.0

    power = np.abs(spectrum) ** 2
    return float(np.mean(power))


def _normalize_lag_steps(lag_steps: list[int] | tuple[int, ...] | None) -> list[int]:
    if lag_steps is None:
        return []
    out: list[int] = []
    for lag in lag_steps:
        lag_int = int(lag)
        if lag_int <= 0:
            continue
        if lag_int not in out:
            out.append(lag_int)
    return sorted(out)


def build_univariate_features(
    series: pd.Series,
    window: int = 60,
    lag_steps: list[int] | tuple[int, ...] | None = None,
    fft_enabled: bool = False,
    fft_include_dc: bool = False,
) -> pd.DataFrame:
    s = series.astype(float)
    feat = pd.DataFrame({
        "value": s,
        "roll_mean": s.rolling(window).mean(),
        "roll_std": s.rolling(window).std(),
        "roll_median": s.rolling(window).median(),
        "delta": s.diff(),
        "roll_min": s.rolling(window).min(),
        "roll_max": s.rolling(window).max(),
    })
    for lag in _normalize_lag_steps(lag_steps):
        feat[f"lag_{lag}"] = s.shift(lag)
    if fft_enabled:
        feat["fft_energy"] = s.rolling(window).apply(
            lambda values: fft_window_energy(values, include_dc=fft_include_dc),
            raw=True,
        )
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat


def build_multivariate_features(
    df: pd.DataFrame,
    window: int = 60,
    lag_steps: list[int] | tuple[int, ...] | None = None,
    fft_enabled: bool = False,
    fft_include_dc: bool = False,
) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).copy()
    lag_steps = _normalize_lag_steps(lag_steps)
    parts = []
    for col in numeric.columns:
        s = numeric[col]
        part = pd.DataFrame({
            f"{col}_mean": s.rolling(window).mean(),
            f"{col}_std": s.rolling(window).std(),
            f"{col}_delta": s.diff(),
        })
        for lag in lag_steps:
            part[f"{col}_lag_{lag}"] = s.shift(lag)
        if fft_enabled:
            part[f"{col}_fft_energy"] = s.rolling(window).apply(
                lambda values: fft_window_energy(values, include_dc=fft_include_dc),
                raw=True,
            )
        parts.append(part)
    feat = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat
