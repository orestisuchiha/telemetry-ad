from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _select_signal(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    preferred_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[str | None, pd.Series]:
    exclude = {timestamp_col, label_col, "anomaly", "Label", "changepoint", "is_anomaly"}
    numeric = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore").select_dtypes(include=["number"])
    if numeric.empty:
        return None, pd.Series(dtype=float)

    for col in preferred_columns or []:
        if col and col in numeric.columns:
            return col, numeric[col].astype(float)

    return str(numeric.columns[0]), numeric.iloc[:, 0].astype(float)


def run_stl_seasonality_analysis(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    report_dir: Path,
    dataset: str,
    variant: str,
    period: int | None,
    enabled: bool = True,
    robust: bool = True,
    preferred_columns: list[str] | tuple[str, ...] | None = None,
) -> dict:
    summary = {
        "enabled": bool(enabled),
        "status": "disabled" if not enabled else "pending",
        "signal_column": None,
        "period": None if period is None else int(period),
        "robust": bool(robust),
    }
    if not enabled:
        return summary

    signal_col, signal = _select_signal(
        df=df,
        timestamp_col=timestamp_col,
        label_col=label_col,
        preferred_columns=preferred_columns,
    )
    if signal_col is None:
        summary["status"] = "skipped_no_numeric_signal"
        return summary

    summary["signal_column"] = signal_col

    if period is None or int(period) < 2:
        summary["status"] = "skipped_invalid_period"
        return summary

    min_required = int(period) * 2
    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    valid = signal.notna() & timestamps.notna()
    signal = signal.loc[valid].reset_index(drop=True)
    timestamps = timestamps.loc[valid].reset_index(drop=True)

    summary["series_length"] = int(len(signal))
    summary["required_length"] = int(min_required)
    if len(signal) < min_required:
        summary["status"] = "skipped_insufficient_length"
        return summary

    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        summary["status"] = "skipped_missing_dependency"
        summary["missing_dependency"] = "statsmodels"
        return summary

    stl = STL(signal.to_numpy(dtype=float), period=int(period), robust=bool(robust))
    result = stl.fit()

    observed = np.asarray(result.observed, dtype=float)
    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(result.seasonal, dtype=float)
    resid = np.asarray(result.resid, dtype=float)

    components = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "observed": observed,
            "trend": trend,
            "seasonal": seasonal,
            "residual": resid,
        }
    )
    components.to_csv(report_dir / "seasonality_components.csv", index=False)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(timestamps, observed, linewidth=1.0)
    axes[0].set_title(f"{dataset}:{variant} STL observed ({signal_col})")
    axes[1].plot(timestamps, trend, linewidth=1.0)
    axes[1].set_ylabel("trend")
    axes[2].plot(timestamps, seasonal, linewidth=1.0)
    axes[2].set_ylabel("seasonal")
    axes[3].plot(timestamps, resid, linewidth=1.0)
    axes[3].set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(report_dir / "seasonality_stl_plot.png", dpi=160)
    plt.close(fig)

    eps = 1e-12
    resid_var = float(np.var(resid))
    trend_strength = max(0.0, 1.0 - resid_var / (float(np.var(resid + trend)) + eps))
    seasonal_strength = max(0.0, 1.0 - resid_var / (float(np.var(resid + seasonal)) + eps))
    resid_abs = np.abs(resid)
    resid_threshold = float(np.quantile(resid_abs, 0.99))
    resid_outliers = int(np.sum(resid_abs >= resid_threshold))

    summary.update(
        {
            "status": "ok",
            "trend_strength": trend_strength,
            "seasonality_strength": seasonal_strength,
            "residual_std": float(np.std(resid)),
            "residual_abs_p99": resid_threshold,
            "residual_outlier_count": resid_outliers,
            "artifacts": {
                "components_csv": "seasonality_components.csv",
                "plot_png": "seasonality_stl_plot.png",
            },
        }
    )
    return summary
