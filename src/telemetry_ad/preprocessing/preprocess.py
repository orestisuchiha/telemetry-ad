import pandas as pd


def basic_preprocess(
    df: pd.DataFrame,
    timestamp_col: str,
    resample_rule: str | None = None,
    ewma_alpha: float | None = None,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    exclude_cols = exclude_cols or []
    out = out.sort_values(timestamp_col).reset_index(drop=True)
    out = out.ffill()

    if resample_rule:
        protected = [c for c in exclude_cols if c in out.columns]
        preserved = out[[timestamp_col] + protected].set_index(timestamp_col).resample(resample_rule).last()
        numeric = out.drop(columns=protected, errors="ignore").set_index(timestamp_col)
        numeric = numeric.resample(resample_rule).mean(numeric_only=True)
        out = numeric.join(preserved, how="left").ffill().reset_index()

    if ewma_alpha is not None:
        numeric_cols = out.select_dtypes(include=["number"]).columns
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        out[numeric_cols] = out[numeric_cols].ewm(alpha=ewma_alpha).mean()

    return out
