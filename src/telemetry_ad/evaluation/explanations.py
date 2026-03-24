from __future__ import annotations

from pathlib import Path

import numpy as np


def _float(value) -> float:
    return float(np.asarray(value, dtype=float))


def _feature_parts(feature_name: str) -> tuple[str, str]:
    if "_lag_" in feature_name:
        source, lag = feature_name.split("_lag_", 1)
        return source, f"lag_{lag}"
    for suffix in ("_fft_energy", "_mean", "_std", "_delta", "_roll_mean", "_roll_std", "_roll_median", "_roll_min", "_roll_max"):
        if feature_name.endswith(suffix):
            source = feature_name[: -len(suffix)]
            return source or feature_name, suffix.removeprefix("_")
    if "_" in feature_name:
        source, stat = feature_name.rsplit("_", 1)
        return source, stat
    return feature_name, feature_name


def _top_feature_entries(
    feature_scores: np.ndarray,
    feature_names: list[str],
    raw_values: np.ndarray | None = None,
    *,
    top_k: int = 3,
    score_label: str = "score",
) -> list[dict]:
    arr = np.asarray(feature_scores, dtype=float).reshape(-1)
    idx = np.argsort(arr)[::-1][:top_k]
    out = []
    raw = None if raw_values is None else np.asarray(raw_values, dtype=float).reshape(-1)
    for i in idx:
        source_metric, statistic = _feature_parts(str(feature_names[i]))
        row = {
            "feature": str(feature_names[i]),
            "source_metric": source_metric,
            "statistic": statistic,
            score_label: _float(arr[i]),
        }
        if raw is not None and i < len(raw):
            row["feature_value"] = _float(raw[i])
        out.append(row)
    return out


def explain_zscore_events(
    events: list[dict],
    scores: np.ndarray,
    feature_z: np.ndarray,
    raw_feature_values: np.ndarray,
    feature_names: list[str],
    timestamps: np.ndarray,
    threshold: float,
    *,
    top_k: int = 3,
) -> list[dict]:
    out = []
    score_arr = np.asarray(scores, dtype=float)
    ts_arr = np.asarray(timestamps)
    for event in events:
        start = int(event["start_idx"])
        end = int(event["end_idx"])
        peak_idx = start + int(np.argmax(score_arr[start : end + 1]))
        top_features = _top_feature_entries(
            feature_z[peak_idx],
            feature_names,
            raw_values=raw_feature_values[peak_idx],
            top_k=top_k,
            score_label="zscore",
        )
        trigger = top_features[0] if top_features else {}
        out.append(
            {
                **event,
                "model": "zscore",
                "peak_idx": peak_idx,
                "peak_ts": str(ts_arr[peak_idx]),
                "peak_score": _float(score_arr[peak_idx]),
                "threshold": float(threshold),
                "trigger_metric": trigger.get("source_metric"),
                "trigger_feature": trigger.get("feature"),
                "trigger_statistic": trigger.get("statistic"),
                "trigger_zscore": trigger.get("zscore"),
                "top_features": top_features,
            }
        )
    return out


def explain_iforest_events(
    events: list[dict],
    scores: np.ndarray,
    scaled_feature_values: np.ndarray,
    raw_feature_values: np.ndarray,
    feature_names: list[str],
    timestamps: np.ndarray,
    threshold: float,
    *,
    top_k: int = 3,
) -> list[dict]:
    out = []
    score_arr = np.asarray(scores, dtype=float)
    ts_arr = np.asarray(timestamps)
    for event in events:
        start = int(event["start_idx"])
        end = int(event["end_idx"])
        peak_idx = start + int(np.argmax(score_arr[start : end + 1]))
        top_features = _top_feature_entries(
            np.abs(scaled_feature_values[peak_idx]),
            feature_names,
            raw_values=raw_feature_values[peak_idx],
            top_k=top_k,
            score_label="standardized_magnitude",
        )
        out.append(
            {
                **event,
                "model": "iforest",
                "explanation_method": "largest absolute standardized engineered features in the flagged window",
                "peak_idx": peak_idx,
                "peak_ts": str(ts_arr[peak_idx]),
                "peak_score": _float(score_arr[peak_idx]),
                "threshold": float(threshold),
                "top_extreme_features": top_features,
            }
        )
    return out


def _top_timestep_entries(timestep_error: np.ndarray, window_size: int, *, top_k: int = 3) -> list[dict]:
    arr = np.asarray(timestep_error, dtype=float).reshape(-1)
    idx = np.argsort(arr)[::-1][:top_k]
    out = []
    for i in idx:
        offset = int(i) - (int(window_size) - 1)
        label = "t" if offset == 0 else f"t{offset}"
        out.append(
            {
                "window_timestep_index": int(i),
                "offset_from_end": offset,
                "relative_label": label,
                "mse": _float(arr[i]),
            }
        )
    return out


def explain_autoencoder_events(
    events: list[dict],
    scores: np.ndarray,
    channel_error: np.ndarray,
    timestep_error: np.ndarray,
    channel_names: list[str],
    timestamps: np.ndarray,
    threshold: float,
    model_name: str,
    window_size: int,
    *,
    top_k: int = 3,
) -> list[dict]:
    out = []
    score_arr = np.asarray(scores, dtype=float)
    ts_arr = np.asarray(timestamps)
    for event in events:
        start = int(event["start_idx"])
        end = int(event["end_idx"])
        peak_idx = start + int(np.argmax(score_arr[start : end + 1]))
        top_channels = _top_feature_entries(
            channel_error[peak_idx],
            channel_names,
            raw_values=None,
            top_k=top_k,
            score_label="channel_error",
        )
        top_timesteps = _top_timestep_entries(timestep_error[peak_idx], window_size=window_size, top_k=top_k)
        out.append(
            {
                **event,
                "model": model_name,
                "peak_idx": peak_idx,
                "peak_ts": str(ts_arr[peak_idx]),
                "peak_score": _float(score_arr[peak_idx]),
                "threshold": float(threshold),
                "top_channels": top_channels,
                "top_timesteps": top_timesteps,
            }
        )
    return out


def build_model_comparison_summary(metrics_payload: dict) -> tuple[dict, str]:
    model_names = [name for name in ("zscore", "iforest", "lstm_ae", "cnn_ae") if name in metrics_payload]
    rows = []
    for name in model_names:
        payload = metrics_payload[name]
        rows.append(
            {
                "model": name,
                "precision": float(payload.get("precision", 0.0)),
                "recall": float(payload.get("recall", 0.0)),
                "f1": float(payload.get("f1", 0.0)),
                "event_f1": None if payload.get("event_f1") is None else float(payload.get("event_f1")),
                "pred_event_count": int(payload.get("pred_event_count", 0)),
            }
        )

    point_best = max(rows, key=lambda row: row["f1"]) if rows else None
    event_best = max(rows, key=lambda row: float("-inf") if row["event_f1"] is None else row["event_f1"]) if rows else None
    precision_best = max(rows, key=lambda row: row["precision"]) if rows else None
    recall_best = max(rows, key=lambda row: row["recall"]) if rows else None

    summary = {
        "models": rows,
        "best_point_f1_model": None if point_best is None else point_best["model"],
        "best_event_f1_model": None if event_best is None else event_best["model"],
        "highest_precision_model": None if precision_best is None else precision_best["model"],
        "highest_recall_model": None if recall_best is None else recall_best["model"],
    }

    tendencies = {
        "zscore": "Best for sharp spikes dominated by one engineered feature because each alert is tied to a single high robust z-score.",
        "iforest": "Best for unusual combinations of engineered features, using the largest standardized feature magnitudes as a lightweight explanation.",
        "lstm_ae": "Best for sustained sequence deviations where error concentrates on a few sensor channels across multiple timesteps.",
        "cnn_ae": "Best for short local temporal motifs, with explanations showing which channels and window positions carry the largest reconstruction error.",
    }

    lines = [
        "# Explainability Summary",
        "",
        "## Metric Snapshot",
        f"- Best point F1: {summary['best_point_f1_model']}",
        f"- Best event F1: {summary['best_event_f1_model']}",
        f"- Highest precision: {summary['highest_precision_model']}",
        f"- Highest recall: {summary['highest_recall_model']}",
        "",
        "## Model Tendencies",
    ]
    for name in model_names:
        payload = metrics_payload[name]
        event_f1 = payload.get("event_f1")
        event_f1_text = "NA" if event_f1 is None else f"{float(event_f1):.4f}"
        lines.append(
            f"- `{name}`: point F1={float(payload.get('f1', 0.0)):.4f}, event F1={event_f1_text}, "
            f"predicted events={int(payload.get('pred_event_count', 0))}. {tendencies.get(name, '')}"
        )

    return summary, "\n".join(lines) + "\n"


def save_model_comparison_summary(report_dir: Path, metrics_payload: dict) -> dict:
    summary, markdown = build_model_comparison_summary(metrics_payload)
    (Path(report_dir) / "model_explainability_summary.md").write_text(markdown, encoding="utf-8")
    return summary
