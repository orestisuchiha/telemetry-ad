from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from telemetry_ad.dataset_io import load_nab_dataset, load_skab_dataset
from telemetry_ad.evaluation.metrics import point_metrics, score_metrics
from telemetry_ad.evaluation.postprocess import event_overlap_metrics, event_type_counts, extract_events
from telemetry_ad.models.iforest import make_iforest
from telemetry_ad.models.zscore import fit_robust_baseline, score_robust_z
from telemetry_ad.preprocessing.features import build_multivariate_features, build_univariate_features
from telemetry_ad.preprocessing.preprocess import basic_preprocess
from telemetry_ad.utils.io import load_pickle, load_yaml, save_json, save_pickle
from telemetry_ad.utils.seed import set_global_seed


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_runtime_config(dataset: str, base_config_path: str) -> dict:
    base_cfg = load_yaml(base_config_path)
    dataset_cfg_path = Path("configs") / f"{dataset}.yaml"
    dataset_cfg = load_yaml(str(dataset_cfg_path))
    return _deep_merge(base_cfg, dataset_cfg)


def _load_bundle(args, cfg):
    if args.dataset == "nab":
        series = args.series or (cfg.get("series") or [None])[0]
        if not series:
            raise ValueError("NAB requires --series or configs/nab.yaml series entry")
        return load_nab_dataset(
            dataset_root=cfg["dataset_root"],
            series=series,
            labels_json=cfg.get("labels_json"),
            train_ratio=float(cfg.get("training", {}).get("train_ratio", 0.7)),
        )

    test_files = cfg.get("test_files", {})
    return load_skab_dataset(
        train_file=cfg["train_file"],
        test_files=test_files,
        separator=cfg.get("separator", ";"),
        split_name=args.split or cfg.get("split_name"),
    )


def _prepare_split(df, timestamp_col: str, label_col: str, cfg: dict):
    pre = cfg.get("preprocessing", {})
    return basic_preprocess(
        df=df,
        timestamp_col=timestamp_col,
        resample_rule=pre.get("resample_rule"),
        ewma_alpha=pre.get("ewma_alpha"),
        exclude_cols=[label_col],
    )


def _feature_frame(df, dataset: str, value_col: str | None, label_col: str, timestamp_col: str, window_size: int):
    if dataset == "nab":
        feat = build_univariate_features(df[value_col], window=window_size)
    else:
        drop_cols = [c for c in (timestamp_col, label_col) if c in df.columns]
        feat = build_multivariate_features(df.drop(columns=drop_cols, errors="ignore"), window=window_size)
    return feat


def _align_tail(df, feat_len: int, col: str | None):
    if feat_len <= 0 or not col or (col not in df.columns):
        return np.asarray([])
    return df[col].to_numpy()[-feat_len:]


def _make_output_dir(base: str, dataset: str, variant: str) -> Path:
    out = Path(base) / dataset / variant
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_report_dir(base: str, dataset: str, variant: str) -> Path:
    out = Path(base) / dataset / variant
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_plot(
    path: Path,
    timestamps: np.ndarray,
    signal: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
    threshold: float,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(timestamps, signal, linewidth=1.0, label="signal")
    if len(labels):
        axes[0].scatter(
            np.asarray(timestamps)[labels.astype(bool)],
            np.asarray(signal)[labels.astype(bool)],
            s=12,
            label="true_anomaly",
            alpha=0.7,
        )
    axes[0].scatter(
        np.asarray(timestamps)[preds.astype(bool)],
        np.asarray(signal)[preds.astype(bool)],
        s=12,
        label="pred_anomaly",
        alpha=0.7,
    )
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")

    axes[1].plot(timestamps, scores, linewidth=1.0, label="anomaly_score")
    axes[1].axhline(threshold, color="red", linestyle="--", linewidth=1.0, label="threshold")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_confusion_plot(path: Path, cm: list[list[int]], title: str) -> None:
    arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fmt_metric(value) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def train_offline(args) -> None:
    cfg = _load_runtime_config(dataset=args.dataset, base_config_path=args.config)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_global_seed(seed)
    bundle = _load_bundle(args=args, cfg=cfg)

    window_size = int(cfg.get("training", {}).get("window_size", 60))
    percentile = float(cfg.get("training", {}).get("threshold_percentile", 99.5))
    variant = bundle.variant
    output_dir = _make_output_dir(args.output_dir, args.dataset, variant)

    train_pre = _prepare_split(bundle.train_df, bundle.timestamp_col, bundle.label_col, cfg=cfg)
    train_feat = _feature_frame(
        df=train_pre,
        dataset=args.dataset,
        value_col=bundle.value_col,
        label_col=bundle.label_col,
        timestamp_col=bundle.timestamp_col,
        window_size=window_size,
    )
    if train_feat.empty:
        raise ValueError("Training feature frame is empty. Lower window_size or check preprocessing.")

    X_train = train_feat.to_numpy(dtype=float)
    use_scaler = bool(cfg.get("preprocessing", {}).get("standardize", True))
    scaler = StandardScaler() if use_scaler else None
    X_train_scaled = scaler.fit_transform(X_train) if scaler is not None else X_train

    z_params = fit_robust_baseline(X_train_scaled)
    z_train_scores = score_robust_z(X_train_scaled, z_params)
    z_threshold = float(np.percentile(z_train_scores, percentile))

    iforest = make_iforest(seed=seed)
    iforest.fit(X_train_scaled)
    if_train_scores = -iforest.score_samples(X_train_scaled)
    if_threshold = float(np.percentile(if_train_scores, percentile))

    save_pickle(scaler, str(output_dir / "scaler.pkl"))
    save_pickle(iforest, str(output_dir / "iforest.pkl"))
    save_pickle(z_params, str(output_dir / "zscore_params.pkl"))
    save_json(
        {
            "zscore": z_threshold,
            "iforest": if_threshold,
            "percentile": percentile,
        },
        str(output_dir / "thresholds.json"),
    )
    save_json(
        {
            "dataset": args.dataset,
            "variant": variant,
            "window_size": window_size,
            "timestamp_col": bundle.timestamp_col,
            "value_col": bundle.value_col,
            "label_col": bundle.label_col,
            "feature_columns": list(train_feat.columns),
        },
        str(output_dir / "metadata.json"),
    )

    print(f"[train] dataset={args.dataset} variant={variant}")
    print(f"[train] output_dir={output_dir}")
    print(f"[train] thresholds zscore={z_threshold:.6f} iforest={if_threshold:.6f}")


def evaluate_offline(args) -> None:
    cfg = _load_runtime_config(dataset=args.dataset, base_config_path=args.config)
    bundle = _load_bundle(args=args, cfg=cfg)
    variant = bundle.variant
    artifact_dir = Path(args.artifacts_dir) / args.dataset / variant
    report_dir = _make_report_dir(args.reports_dir, args.dataset, variant)

    scaler = load_pickle(str(artifact_dir / "scaler.pkl"))
    iforest = load_pickle(str(artifact_dir / "iforest.pkl"))
    z_params = load_pickle(str(artifact_dir / "zscore_params.pkl"))
    with (artifact_dir / "thresholds.json").open("r", encoding="utf-8") as f:
        thresholds = json.load(f)
    with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    window_size = int(metadata["window_size"])
    test_pre = _prepare_split(bundle.test_df, bundle.timestamp_col, bundle.label_col, cfg=cfg)
    test_feat = _feature_frame(
        df=test_pre,
        dataset=args.dataset,
        value_col=bundle.value_col,
        label_col=bundle.label_col,
        timestamp_col=bundle.timestamp_col,
        window_size=window_size,
    )
    if test_feat.empty:
        raise ValueError("Test feature frame is empty. Lower window_size or check preprocessing.")

    feature_columns = metadata.get("feature_columns", list(test_feat.columns))
    test_feat = test_feat.reindex(columns=feature_columns, fill_value=0.0)

    X_test = test_feat.to_numpy(dtype=float)
    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test

    y_true = _align_tail(test_pre, len(test_feat), bundle.label_col).astype(int)
    ts = _align_tail(test_pre, len(test_feat), bundle.timestamp_col)
    if bundle.value_col and bundle.value_col in test_pre.columns:
        signal = _align_tail(test_pre, len(test_feat), bundle.value_col).astype(float)
    else:
        drop_cols = [c for c in (bundle.timestamp_col, bundle.label_col) if c in test_pre.columns]
        numeric = test_pre.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number"])
        signal_col = numeric.columns[0] if len(numeric.columns) else None
        signal = _align_tail(test_pre, len(test_feat), signal_col).astype(float) if signal_col else np.zeros(len(test_feat))

    z_scores = score_robust_z(X_test_scaled, z_params)
    z_pred = (z_scores > float(thresholds["zscore"])).astype(int)
    if_scores = -iforest.score_samples(X_test_scaled)
    if_pred = (if_scores > float(thresholds["iforest"])).astype(int)

    min_collective = int(cfg.get("inference", {}).get("alert_min_segment_length", 3))
    true_events = extract_events(y_true, ts, min_collective=min_collective)
    z_events = extract_events(z_pred, ts, min_collective=min_collective)
    if_events = extract_events(if_pred, ts, min_collective=min_collective)

    z_metrics = point_metrics(y_true, z_pred)
    if_metrics = point_metrics(y_true, if_pred)
    z_rank_metrics = score_metrics(y_true, z_scores)
    if_rank_metrics = score_metrics(y_true, if_scores)
    z_event_metrics = event_overlap_metrics(true_events, z_events)
    if_event_metrics = event_overlap_metrics(true_events, if_events)
    z_cm = confusion_matrix(y_true, z_pred, labels=[0, 1]).tolist()
    if_cm = confusion_matrix(y_true, if_pred, labels=[0, 1]).tolist()

    save_json(true_events, str(report_dir / "events_true.json"))
    save_json(z_events, str(report_dir / "events_zscore.json"))
    save_json(if_events, str(report_dir / "events_iforest.json"))

    save_json(
        {
            "dataset": args.dataset,
            "variant": variant,
            "settings": {"min_collective_event_length": min_collective},
            "ground_truth": {
                "event_count": len(true_events),
                "event_type_counts": event_type_counts(true_events),
            },
            "zscore": {
                **z_metrics,
                **z_rank_metrics,
                **z_event_metrics,
                "pred_event_count": len(z_events),
                "pred_event_type_counts": event_type_counts(z_events),
                "confusion_matrix": z_cm,
            },
            "iforest": {
                **if_metrics,
                **if_rank_metrics,
                **if_event_metrics,
                "pred_event_count": len(if_events),
                "pred_event_type_counts": event_type_counts(if_events),
                "confusion_matrix": if_cm,
            },
        },
        str(report_dir / "metrics.json"),
    )

    pred_df = pd.DataFrame(
        {
            "timestamp": ts,
            "y_true": y_true,
            "zscore_score": z_scores,
            "zscore_pred": z_pred,
            "iforest_score": if_scores,
            "iforest_pred": if_pred,
        }
    )
    pred_df.to_csv(report_dir / "predictions.csv", index=False)

    _save_plot(
        path=report_dir / "zscore_plot.png",
        timestamps=ts,
        signal=signal,
        labels=y_true,
        scores=z_scores,
        preds=z_pred,
        threshold=float(thresholds["zscore"]),
        title=f"{args.dataset}:{variant} Z-score",
    )
    _save_plot(
        path=report_dir / "iforest_plot.png",
        timestamps=ts,
        signal=signal,
        labels=y_true,
        scores=if_scores,
        preds=if_pred,
        threshold=float(thresholds["iforest"]),
        title=f"{args.dataset}:{variant} Isolation Forest",
    )
    _save_confusion_plot(
        path=report_dir / "zscore_confusion_matrix.png",
        cm=z_cm,
        title=f"{args.dataset}:{variant} Z-score CM",
    )
    _save_confusion_plot(
        path=report_dir / "iforest_confusion_matrix.png",
        cm=if_cm,
        title=f"{args.dataset}:{variant} IF CM",
    )

    print(f"[eval] dataset={args.dataset} variant={variant}")
    print(f"[eval] reports={report_dir}")
    print(f"[eval] zscore_f1={z_metrics['f1']:.4f} iforest_f1={if_metrics['f1']:.4f}")
    print(
        f"[eval] zscore_event_f1={_fmt_metric(z_event_metrics['event_f1'])} "
        f"iforest_event_f1={_fmt_metric(if_event_metrics['event_f1'])}"
    )


def infer_stream_pi(args) -> None:
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"[infer] dataset={args.dataset} model={args.model} series={args.series} split={args.split}")
    print("[infer] Person B streaming path not implemented in orchestration yet.")
