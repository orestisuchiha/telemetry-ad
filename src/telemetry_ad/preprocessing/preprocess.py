import pandas as pd


def _resolve_interpolation_method(interpolate: bool | str | None) -> str | None:
    if isinstance(interpolate, str):
        value = interpolate.strip().lower()
        if value in {"", "false", "none", "off", "no", "0"}:
            return None
        if value in {"true", "on", "yes", "1"}:
            return "linear"
        return interpolate
    if interpolate:
        return "linear"
    return None


def basic_preprocess(
    df: pd.DataFrame,
    timestamp_col: str,
    resample_rule: str | None = None,
    forward_fill: bool = True,
    interpolate: bool | str | None = None,
    interpolate_limit_direction: str = "both",
    rolling_detrend_window: int | None = None,
    ewma_alpha: float | None = None,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    exclude_cols = exclude_cols or []
    out = out.sort_values(timestamp_col).reset_index(drop=True)

    if resample_rule:
        protected = [c for c in exclude_cols if c in out.columns]
        preserved = out[[timestamp_col] + protected].set_index(timestamp_col).resample(resample_rule).last()
        numeric = out.drop(columns=protected, errors="ignore").set_index(timestamp_col)
        numeric = numeric.resample(resample_rule).mean(numeric_only=True)
        out = numeric.join(preserved, how="left").reset_index()

    numeric_cols = out.select_dtypes(include=["number"]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    interpolation_method = _resolve_interpolation_method(interpolate)
    if interpolation_method is not None and numeric_cols:
        if interpolation_method == "time":
            indexed = out.set_index(timestamp_col)
            indexed[numeric_cols] = indexed[numeric_cols].interpolate(
                method="time",
                limit_direction=interpolate_limit_direction,
            )
            out = indexed.reset_index()
        else:
            out[numeric_cols] = out[numeric_cols].interpolate(
                method=interpolation_method,
                limit_direction=interpolate_limit_direction,
            )

    if forward_fill:
        out = out.ffill()

    if rolling_detrend_window is not None and numeric_cols:
        window = int(rolling_detrend_window)
        if window > 1:
            trend = out[numeric_cols].rolling(window=window, min_periods=1).mean()
            out[numeric_cols] = out[numeric_cols] - trend

    if ewma_alpha is not None:
        out[numeric_cols] = out[numeric_cols].ewm(alpha=ewma_alpha).mean()

    return out
