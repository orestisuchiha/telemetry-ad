from __future__ import annotations

from pathlib import Path

import numpy as np

from telemetry_ad.evaluation.metrics import point_metrics
from telemetry_ad.evaluation.postprocess import event_overlap_metrics, extract_events


def _score_summary(scores: np.ndarray) -> dict:
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
    }


def _binary_predictions(scores: np.ndarray, threshold: float, suppressed_windows: int = 0) -> np.ndarray:
    pred = (np.asarray(scores, dtype=float) > float(threshold)).astype(int)
    if suppressed_windows > 0:
        pred[:suppressed_windows] = 0
    return pred


def _segment_summaries(scores: np.ndarray, segment_count: int = 3) -> list[dict]:
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return []
    segment_count = max(1, int(segment_count))
    splits = np.array_split(np.arange(len(arr)), segment_count)
    out = []
    for idx, indices in enumerate(splits):
        if len(indices) == 0:
            continue
        segment_scores = arr[indices]
        label = ["early", "mid", "late"][idx] if segment_count == 3 and idx < 3 else f"segment_{idx + 1}"
        out.append(
            {
                "segment": label,
                "start_idx": int(indices[0]),
                "end_idx": int(indices[-1]),
                **_score_summary(segment_scores),
            }
        )
    return out


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0.0):
        return None
    return float(num) / float(den)


def _shift_level(mean_shift_std: float | None, p95_ratio: float | None) -> str:
    if mean_shift_std is None or p95_ratio is None:
        return "unknown"
    if abs(mean_shift_std) >= 1.0 or p95_ratio >= 1.5:
        return "clear_shift"
    if abs(mean_shift_std) >= 0.5 or p95_ratio >= 1.2:
        return "moderate_shift"
    return "limited_shift"


def _recalibration_effect_label(artifact_f1: float, calibrated_f1: float) -> str:
    delta = float(calibrated_f1) - float(artifact_f1)
    if delta > 0.02:
        return "recalibration_helped"
    if delta < -0.02:
        return "recalibration_hurt"
    return "recalibration_neutral"


def build_nonstationarity_report(
    *,
    model_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    artifact_threshold: float,
    calibrated_threshold: float,
    calibrated_suppressed_windows: int,
    min_collective: int,
    training_threshold_summary: dict | None = None,
    warmup_windows: int = 0,
    segment_count: int = 3,
) -> dict:
    arr = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    ts = np.asarray(timestamps)

    artifact_pred = _binary_predictions(arr, artifact_threshold, suppressed_windows=0)
    calibrated_pred = _binary_predictions(
        arr,
        calibrated_threshold,
        suppressed_windows=int(calibrated_suppressed_windows),
    )

    true_events = extract_events(y_true, ts, min_collective=min_collective)
    artifact_events = extract_events(artifact_pred, ts, min_collective=min_collective)
    calibrated_events = extract_events(calibrated_pred, ts, min_collective=min_collective)

    artifact_metrics = {
        **point_metrics(y_true, artifact_pred),
        **event_overlap_metrics(true_events, artifact_events),
        "alert_count": int(artifact_pred.sum()),
        "pred_event_count": int(len(artifact_events)),
    }
    calibrated_metrics = {
        **point_metrics(y_true, calibrated_pred),
        **event_overlap_metrics(true_events, calibrated_events),
        "alert_count": int(calibrated_pred.sum()),
        "pred_event_count": int(len(calibrated_events)),
    }

    overall_summary = _score_summary(arr)
    warmup_summary = _score_summary(arr[:warmup_windows]) if warmup_windows > 0 else None
    training_score_summary = None if not training_threshold_summary else training_threshold_summary.get("train_score_summary")
    training_mean = None if not training_score_summary else training_score_summary.get("mean")
    training_std = None if not training_score_summary else training_score_summary.get("std")
    training_p95 = None if not training_score_summary else training_score_summary.get("p95")

    warmup_mean_shift_std = None
    overall_mean_shift_std = None
    warmup_p95_ratio = None
    overall_p95_ratio = None
    if training_mean is not None and training_std not in (None, 0.0):
        overall_mean_shift_std = (float(overall_summary["mean"]) - float(training_mean)) / float(training_std)
        if warmup_summary is not None and warmup_summary.get("count", 0) > 0:
            warmup_mean_shift_std = (float(warmup_summary["mean"]) - float(training_mean)) / float(training_std)
    if training_p95 not in (None, 0.0):
        overall_p95_ratio = _safe_ratio(overall_summary.get("p95"), training_p95)
        if warmup_summary is not None and warmup_summary.get("count", 0) > 0:
            warmup_p95_ratio = _safe_ratio(warmup_summary.get("p95"), training_p95)

    return {
        "model": model_name,
        "method": "compare fixed artifact threshold vs calibrated threshold and summarize score-distribution shift over time",
        "training_threshold": training_threshold_summary,
        "artifact_fixed": {
            "threshold": float(artifact_threshold),
            **artifact_metrics,
        },
        "calibrated_stream": {
            "threshold": float(calibrated_threshold),
            "suppressed_windows": int(calibrated_suppressed_windows),
            **calibrated_metrics,
        },
        "delta": {
            "threshold": float(calibrated_threshold) - float(artifact_threshold),
            "f1": float(calibrated_metrics["f1"]) - float(artifact_metrics["f1"]),
            "event_f1": None
            if artifact_metrics["event_f1"] is None or calibrated_metrics["event_f1"] is None
            else float(calibrated_metrics["event_f1"]) - float(artifact_metrics["event_f1"]),
            "alert_count": int(calibrated_metrics["alert_count"]) - int(artifact_metrics["alert_count"]),
        },
        "score_distribution": {
            "overall": overall_summary,
            "warmup": warmup_summary,
            "segments": _segment_summaries(arr, segment_count=segment_count),
        },
        "shift_indicators": {
            "warmup_mean_shift_std": warmup_mean_shift_std,
            "overall_mean_shift_std": overall_mean_shift_std,
            "warmup_p95_ratio_vs_train": warmup_p95_ratio,
            "overall_p95_ratio_vs_train": overall_p95_ratio,
            "shift_level": _shift_level(overall_mean_shift_std, overall_p95_ratio),
            "recalibration_effect": _recalibration_effect_label(
                artifact_f1=float(artifact_metrics["f1"]),
                calibrated_f1=float(calibrated_metrics["f1"]),
            ),
        },
    }


def build_nonstationarity_summary(report_by_model: dict) -> tuple[dict, str]:
    summary = {"method": "fixed_vs_calibrated_threshold_comparison", "models": report_by_model}
    lines = [
        "# Non-Stationarity Summary",
        "",
        "This section compares fixed artifact thresholds against the calibrated deployment thresholds and summarizes how anomaly-score distributions shift across the evaluation horizon.",
        "",
    ]
    for model_name, payload in report_by_model.items():
        artifact = payload["artifact_fixed"]
        calibrated = payload["calibrated_stream"]
        shift = payload["shift_indicators"]
        lines.append(f"## {model_name}")
        lines.append(
            f"- Artifact threshold F1={float(artifact['f1']):.4f}, alerts={int(artifact['alert_count'])}; "
            f"calibrated F1={float(calibrated['f1']):.4f}, alerts={int(calibrated['alert_count'])}."
        )
        lines.append(
            f"- Score shift level: `{shift['shift_level']}`; recalibration effect: `{shift['recalibration_effect']}`."
        )
        warmup_ratio = shift.get("warmup_p95_ratio_vs_train")
        overall_ratio = shift.get("overall_p95_ratio_vs_train")
        lines.append(
            f"- Train-to-test p95 ratio: warmup={('NA' if warmup_ratio is None else f'{float(warmup_ratio):.3f}')}, "
            f"overall={('NA' if overall_ratio is None else f'{float(overall_ratio):.3f}')}"
        )
        lines.append("")
    return summary, "\n".join(lines)


def save_nonstationarity_summary(report_dir: Path, report_by_model: dict) -> dict:
    summary, markdown = build_nonstationarity_summary(report_by_model)
    (Path(report_dir) / "nonstationarity_summary.md").write_text(markdown, encoding="utf-8")
    return summary
