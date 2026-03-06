import numpy as np
import pandas as pd


def build_univariate_features(series: pd.Series, window: int = 60) -> pd.DataFrame:
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
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat


def build_multivariate_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).copy()
    parts = []
    for col in numeric.columns:
        s = numeric[col]
        part = pd.DataFrame({
            f"{col}_mean": s.rolling(window).mean(),
            f"{col}_std": s.rolling(window).std(),
            f"{col}_delta": s.diff(),
        })
        parts.append(part)
    feat = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat
