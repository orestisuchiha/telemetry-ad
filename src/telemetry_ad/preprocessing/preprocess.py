import pandas as pd


def basic_preprocess(
    df: pd.DataFrame,
    timestamp_col: str,
    resample_rule: str | None = None,
    ewma_alpha: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(timestamp_col).reset_index(drop=True)
    out = out.ffill()

    if resample_rule:
        out = out.set_index(timestamp_col).resample(resample_rule).mean(numeric_only=True).ffill().reset_index()

    if ewma_alpha is not None:
        numeric_cols = out.select_dtypes(include=["number"]).columns
        out[numeric_cols] = out[numeric_cols].ewm(alpha=ewma_alpha).mean()

    return out
