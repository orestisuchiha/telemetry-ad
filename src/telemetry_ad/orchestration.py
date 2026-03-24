from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from telemetry_ad.dataset_io import load_nab_dataset, load_skab_dataset
from telemetry_ad.evaluation.metrics import point_metrics, score_metrics
from telemetry_ad.evaluation.postprocess import event_overlap_metrics, event_type_counts, extract_events
from telemetry_ad.models.ae_utils import make_windows, reconstruction_scores, train_cnn_autoencoder, train_lstm_autoencoder
from telemetry_ad.models.cnn_ae import CNNAutoencoder
from telemetry_ad.models.iforest import make_iforest
from telemetry_ad.models.lstm_ae import LSTMAutoencoder
from telemetry_ad.models.zscore import fit_robust_baseline, score_robust_z
from telemetry_ad.preprocessing.features import build_multivariate_features, build_univariate_features
from telemetry_ad.preprocessing.preprocess import basic_preprocess
from telemetry_ad.streaming.alerting import append_alert
from telemetry_ad.streaming.stream import RingBuffer
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
        forward_fill=bool(pre.get("forward_fill", True)),
        interpolate=pre.get("interpolate"),
        interpolate_limit_direction=str(pre.get("interpolate_limit_direction", "both")),
        ewma_alpha=pre.get("ewma_alpha"),
        exclude_cols=[label_col],
    )


def _lag_steps(cfg: dict, window_size: int) -> list[int]:
    feat_cfg = cfg.get("feature_engineering", {})
    raw_steps = feat_cfg.get("lag_steps") or []
    steps = sorted({int(step) for step in raw_steps if int(step) > 0})
    if steps and max(steps) >= window_size:
        raise ValueError("feature_engineering.lag_steps must be smaller than training.window_size")
    return steps


def _feature_frame(
    df,
    dataset: str,
    value_col: str | None,
    label_col: str,
    timestamp_col: str,
    window_size: int,
    cfg: dict,
):
    lag_steps = _lag_steps(cfg=cfg, window_size=window_size)
    if dataset == "nab":
        feat = build_univariate_features(df[value_col], window=window_size, lag_steps=lag_steps)
    else:
        exclude = {timestamp_col, label_col, "anomaly", "Label", "changepoint", "is_anomaly"}
        drop_cols = [c for c in exclude if c in df.columns]
        feat = build_multivariate_features(
            df.drop(columns=drop_cols, errors="ignore"),
            window=window_size,
            lag_steps=lag_steps,
        )
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


def _advanced_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    exclude = {timestamp_col, label_col, "anomaly", "Label", "changepoint", "is_anomaly"}
    drop_cols = [c for c in exclude if c in df.columns]
    numeric = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number"])
    if feature_columns is not None:
        numeric = numeric.reindex(columns=feature_columns, fill_value=0.0)
    return numeric


def _advanced_matrix(
    df: pd.DataFrame,
    timestamp_col: str,
    label_col: str,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    return _advanced_frame(
        df=df,
        timestamp_col=timestamp_col,
        label_col=label_col,
        feature_columns=feature_columns,
    ).to_numpy(dtype=float)


def _window_labels(binary_labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    labels = np.asarray(binary_labels).astype(int)
    wins = make_windows(labels.reshape(-1, 1), window_size=window_size, stride=stride)
    if len(wins) == 0:
        return np.asarray([], dtype=int)
    return (wins.max(axis=1).reshape(-1) > 0).astype(int)


def _window_end_values(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    arr = np.asarray(values)
    if len(arr) < window_size:
        return np.asarray([])
    idx = np.arange(window_size - 1, len(arr), stride)
    return arr[idx]


def _stream_feature_vector(window_arr: np.ndarray, dataset: str, lag_steps: list[int] | None = None) -> np.ndarray:
    lag_steps = sorted({int(step) for step in (lag_steps or []) if int(step) > 0})
    if dataset == "nab":
        s = window_arr[:, 0]
        std = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
        delta = float(s[-1] - s[-2]) if len(s) > 1 else 0.0
        feats = [float(s[-1]), float(np.mean(s)), std, float(np.median(s)), delta, float(np.min(s)), float(np.max(s))]
        feats.extend(float(s[-1 - lag]) for lag in lag_steps)
        return np.asarray([feats], dtype=float)

    feats = []
    for c in range(window_arr.shape[1]):
        s = window_arr[:, c]
        std = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
        delta = float(s[-1] - s[-2]) if len(s) > 1 else 0.0
        feats.extend([float(np.mean(s)), std, delta])
        feats.extend(float(s[-1 - lag]) for lag in lag_steps)
    return np.asarray([feats], dtype=float)


def _resolve_threshold_calibration(cfg: dict, model_name: str) -> dict:
    infer_cfg = cfg.get("inference", {})
    cal_cfg = infer_cfg.get("threshold_calibration", {})
    enabled = bool(cal_cfg.get("enabled", False))
    default_cfg = cal_cfg.get("default", {})
    model_cfg = (cal_cfg.get("models") or {}).get(model_name, {})
    merged = _deep_merge(default_cfg, model_cfg)
    return {
        "enabled": enabled,
        "mode": str(merged.get("mode", "artifact")),
        "percentile": float(merged.get("percentile", 99.5)),
        "warmup_windows": int(cal_cfg.get("warmup_windows", 0)),
        "suppress_during_warmup": bool(cal_cfg.get("suppress_during_warmup", True)),
    }


def _calibrate_threshold_from_scores(
    scores: np.ndarray,
    base_threshold: float,
    cfg: dict,
    model_name: str,
) -> tuple[float, int, dict]:
    cal = _resolve_threshold_calibration(cfg=cfg, model_name=model_name)
    payload = {
        "enabled": cal["enabled"],
        "mode": cal["mode"],
        "base_threshold": float(base_threshold),
        "effective_threshold": float(base_threshold),
        "percentile": None if cal["mode"] == "artifact" else float(cal["percentile"]),
        "warmup_windows": 0,
        "suppressed_windows": 0,
    }
    if not cal["enabled"] or cal["mode"] == "artifact":
        return float(base_threshold), 0, payload

    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return float(base_threshold), 0, payload

    warmup = min(int(cal["warmup_windows"]), len(arr))
    if warmup <= 0:
        return float(base_threshold), 0, payload

    threshold = float(np.percentile(arr[:warmup], cal["percentile"]))
    suppressed = warmup if cal["suppress_during_warmup"] else 0
    payload.update(
        {
            "effective_threshold": threshold,
            "warmup_windows": warmup,
            "suppressed_windows": suppressed,
        }
    )
    return threshold, suppressed, payload


def _binary_predictions(scores: np.ndarray, threshold: float, suppressed_windows: int = 0) -> np.ndarray:
    pred = (np.asarray(scores, dtype=float) > float(threshold)).astype(int)
    if suppressed_windows > 0:
        pred[:suppressed_windows] = 0
    return pred


def _resolve_variant(args, cfg: dict) -> str:
    if args.dataset == "nab":
        series = args.series or (cfg.get("series") or [None])[0]
        if not series:
            raise ValueError("NAB requires --series or configs/nab.yaml series entry")
        return series
    return args.split or cfg.get("split_name") or "anomalyfree_vs_valve1_1"


def _iter_local_stream_rows(bundle, cfg: dict, metadata: dict):
    test_pre = _prepare_split(bundle.test_df, bundle.timestamp_col, bundle.label_col, cfg=cfg)
    adv_columns = metadata.get("advanced_feature_columns") or None
    raw_test = _advanced_matrix(
        test_pre,
        bundle.timestamp_col,
        bundle.label_col,
        feature_columns=adv_columns,
    )
    timestamps = test_pre[bundle.timestamp_col].astype(str).to_numpy()
    for timestamp, row in zip(timestamps, raw_test, strict=False):
        yield str(timestamp), np.asarray(row, dtype=float)


def _fetch_json(url: str, timeout: float) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _iter_api_stream_rows(args, metadata: dict):
    base_url = (getattr(args, "api_base_url", None) or "").rstrip("/")
    if not base_url:
        raise ValueError("API mode requires --api-base-url")

    timeout = float(getattr(args, "api_timeout", 10.0))
    batch_size = int(getattr(args, "api_batch_size", 1))
    cursor = int(getattr(args, "api_start_cursor", 0))
    if batch_size < 1:
        raise ValueError("--api-batch-size must be >= 1")
    if cursor < 0:
        raise ValueError("--api-start-cursor must be >= 0")

    health = _fetch_json(f"{base_url}/health", timeout=timeout)
    api_dataset = health.get("dataset")
    if api_dataset and api_dataset != args.dataset:
        raise ValueError(f"API dataset mismatch: expected {args.dataset}, got {api_dataset}")

    value_columns = list(health.get("value_columns") or [])
    if not value_columns:
        raise ValueError("API health response did not include any value_columns")

    expected_columns = metadata.get("advanced_feature_columns") or value_columns
    missing = [col for col in expected_columns if col not in value_columns]
    if missing:
        raise ValueError(
            "API stream is missing required value columns: "
            + ", ".join(missing)
        )

    while True:
        query = urlencode({"cursor": cursor, "batch_size": batch_size})
        payload = _fetch_json(f"{base_url}/stream/next?{query}", timeout=timeout)
        rows = payload.get("rows") or []
        for row in rows:
            values = row.get("values") or {}
            ordered = []
            for col in expected_columns:
                if col not in values:
                    raise ValueError(f"API row is missing required value column: {col}")
                ordered.append(float(values[col]))
            yield str(row.get("timestamp", "")), np.asarray(ordered, dtype=float)

        next_cursor = int(payload.get("next_cursor", cursor + len(rows)))
        done = bool(payload.get("done", False))
        if done:
            break
        if not rows and next_cursor <= cursor:
            break
        cursor = next_cursor


def _score_stream_window(
    model_name: str,
    dataset: str,
    window_arr: np.ndarray,
    cfg: dict,
    scaler,
    z_params,
    iforest,
    seq_scaler,
    ae_model,
    device: str,
) -> float:
    lag_steps = _lag_steps(cfg=cfg, window_size=window_arr.shape[0])
    if model_name == "zscore":
        feat = _stream_feature_vector(window_arr, dataset, lag_steps=lag_steps)
        x = scaler.transform(feat) if scaler is not None else feat
        return float(score_robust_z(x, z_params)[0])

    if model_name == "iforest":
        feat = _stream_feature_vector(window_arr, dataset, lag_steps=lag_steps)
        x = scaler.transform(feat) if scaler is not None else feat
        return float(-iforest.score_samples(x)[0])

    scaled = seq_scaler.transform(window_arr)
    with torch.no_grad():
        t = torch.tensor(scaled[None, ...], dtype=torch.float32, device=device)
        if model_name == "cnn_ae":
            inp = t.permute(0, 2, 1)
            recon = ae_model(inp)
            return float(((recon - inp) ** 2).mean().item())
        recon = ae_model(t)
        return float(((recon - t) ** 2).mean().item())


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
        cfg=cfg,
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
    thresholds_payload = {
        "zscore": z_threshold,
        "iforest": if_threshold,
        "percentile": percentile,
    }

    adv_cfg = cfg.get("advanced", {})
    adv_enabled = bool(adv_cfg.get("enabled", True))
    if adv_enabled:
        device = str(adv_cfg.get("device", "cpu"))
        epochs = int(adv_cfg.get("epochs", 8))
        batch_size = int(adv_cfg.get("batch_size", 128))
        lr = float(adv_cfg.get("lr", 1e-3))
        hidden_dim = int(adv_cfg.get("lstm_hidden_dim", 32))

        adv_train_frame = _advanced_frame(train_pre, bundle.timestamp_col, bundle.label_col)
        raw_train = adv_train_frame.to_numpy(dtype=float)
        if len(raw_train) >= window_size:
            seq_scaler = StandardScaler()
            raw_train_scaled = seq_scaler.fit_transform(raw_train)
            train_windows = make_windows(raw_train_scaled, window_size=window_size, stride=int(cfg.get("training", {}).get("stride", 1)))

            if len(train_windows):
                lstm_ae = train_lstm_autoencoder(
                    windows=train_windows,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )
                cnn_ae = train_cnn_autoencoder(
                    windows=train_windows,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )

                lstm_scores = reconstruction_scores(lstm_ae, train_windows, device=device, model_type="lstm")
                cnn_scores = reconstruction_scores(cnn_ae, train_windows, device=device, model_type="cnn")
                lstm_threshold = float(np.percentile(lstm_scores, percentile))
                cnn_threshold = float(np.percentile(cnn_scores, percentile))

                torch.save(
                    {"state_dict": lstm_ae.state_dict(), "input_dim": train_windows.shape[-1], "hidden_dim": hidden_dim},
                    output_dir / "lstm_ae.pt",
                )
                torch.save(
                    {"state_dict": cnn_ae.state_dict(), "channels": train_windows.shape[-1]},
                    output_dir / "cnn_ae.pt",
                )
                save_pickle(seq_scaler, str(output_dir / "seq_scaler.pkl"))

                thresholds_payload["lstm_ae"] = lstm_threshold
                thresholds_payload["cnn_ae"] = cnn_threshold

    save_json(
        thresholds_payload,
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
            "advanced_enabled": adv_enabled,
            "advanced_feature_columns": list(adv_train_frame.columns) if adv_enabled else [],
        },
        str(output_dir / "metadata.json"),
    )

    print(f"[train] dataset={args.dataset} variant={variant}")
    print(f"[train] output_dir={output_dir}")
    print(f"[train] thresholds zscore={z_threshold:.6f} iforest={if_threshold:.6f}")
    if "lstm_ae" in thresholds_payload and "cnn_ae" in thresholds_payload:
        print(f"[train] thresholds lstm_ae={thresholds_payload['lstm_ae']:.6f} cnn_ae={thresholds_payload['cnn_ae']:.6f}")


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
        cfg=cfg,
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
        signal_source = test_pre[bundle.value_col].to_numpy(dtype=float)
    else:
        drop_cols = [c for c in (bundle.timestamp_col, bundle.label_col) if c in test_pre.columns]
        numeric = test_pre.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number"])
        signal_col = numeric.columns[0] if len(numeric.columns) else None
        signal_source = test_pre[signal_col].to_numpy(dtype=float) if signal_col else np.zeros(len(test_pre))
    signal = signal_source[-len(test_feat):] if len(test_feat) else np.zeros(0)

    z_scores = score_robust_z(X_test_scaled, z_params)
    if_scores = -iforest.score_samples(X_test_scaled)
    z_threshold, z_suppressed, z_cal = _calibrate_threshold_from_scores(
        scores=z_scores,
        base_threshold=float(thresholds["zscore"]),
        cfg=cfg,
        model_name="zscore",
    )
    if_threshold, if_suppressed, if_cal = _calibrate_threshold_from_scores(
        scores=if_scores,
        base_threshold=float(thresholds["iforest"]),
        cfg=cfg,
        model_name="iforest",
    )
    z_pred = _binary_predictions(z_scores, z_threshold, suppressed_windows=z_suppressed)
    if_pred = _binary_predictions(if_scores, if_threshold, suppressed_windows=if_suppressed)

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

    metrics_payload = {
        "dataset": args.dataset,
        "variant": variant,
        "settings": {"min_collective_event_length": min_collective},
        "ground_truth": {
            "event_count": len(true_events),
            "event_type_counts": event_type_counts(true_events),
        },
        "zscore": {
            "threshold": z_threshold,
            "threshold_calibration": z_cal,
            **z_metrics,
            **z_rank_metrics,
            **z_event_metrics,
            "pred_event_count": len(z_events),
            "pred_event_type_counts": event_type_counts(z_events),
            "confusion_matrix": z_cm,
        },
        "iforest": {
            "threshold": if_threshold,
            "threshold_calibration": if_cal,
            **if_metrics,
            **if_rank_metrics,
            **if_event_metrics,
            "pred_event_count": len(if_events),
            "pred_event_type_counts": event_type_counts(if_events),
            "confusion_matrix": if_cm,
        },
    }

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

    # Advanced models are evaluated on raw sequence windows (end-of-window timestamps/labels).
    seq_scaler_path = artifact_dir / "seq_scaler.pkl"
    stride = int(cfg.get("training", {}).get("stride", 1))
    adv_rows = None
    if seq_scaler_path.exists():
        adv_columns = metadata.get("advanced_feature_columns") or None
        seq_scaler = load_pickle(str(seq_scaler_path))
        raw_test = _advanced_matrix(
            test_pre,
            bundle.timestamp_col,
            bundle.label_col,
            feature_columns=adv_columns,
        )
        if len(raw_test) >= window_size:
            raw_test_scaled = seq_scaler.transform(raw_test)
            test_windows = make_windows(raw_test_scaled, window_size=window_size, stride=stride)
            y_true_adv = _window_labels(test_pre[bundle.label_col].to_numpy(), window_size=window_size, stride=stride)
            ts_adv = _window_end_values(test_pre[bundle.timestamp_col].to_numpy(), window_size=window_size, stride=stride)
            signal_adv = _window_end_values(signal_source, window_size=window_size, stride=stride)

            if len(test_windows):
                adv_rows = {"timestamp": ts_adv, "y_true": y_true_adv}
                device = str(cfg.get("advanced", {}).get("device", "cpu"))
                advanced_specs = [
                    ("lstm_ae", artifact_dir / "lstm_ae.pt", "lstm"),
                    ("cnn_ae", artifact_dir / "cnn_ae.pt", "cnn"),
                ]
                for model_name, model_path, model_type in advanced_specs:
                    if not model_path.exists() or model_name not in thresholds:
                        continue
                    checkpoint = torch.load(model_path, map_location=device)
                    if model_name == "lstm_ae":
                        model = LSTMAutoencoder(
                            input_dim=int(checkpoint["input_dim"]),
                            hidden_dim=int(checkpoint.get("hidden_dim", 32)),
                        )
                    else:
                        model = CNNAutoencoder(channels=int(checkpoint["channels"]))
                    model.load_state_dict(checkpoint["state_dict"])

                    adv_scores = reconstruction_scores(model, test_windows, device=device, model_type=model_type)
                    adv_threshold, adv_suppressed, adv_cal = _calibrate_threshold_from_scores(
                        scores=adv_scores,
                        base_threshold=float(thresholds[model_name]),
                        cfg=cfg,
                        model_name=model_name,
                    )
                    adv_pred = _binary_predictions(adv_scores, adv_threshold, suppressed_windows=adv_suppressed)
                    adv_events = extract_events(adv_pred, ts_adv, min_collective=min_collective)
                    adv_metrics = point_metrics(y_true_adv, adv_pred)
                    adv_rank_metrics = score_metrics(y_true_adv, adv_scores)
                    adv_event_metrics = event_overlap_metrics(extract_events(y_true_adv, ts_adv, min_collective=min_collective), adv_events)
                    adv_cm = confusion_matrix(y_true_adv, adv_pred, labels=[0, 1]).tolist()

                    metrics_payload[model_name] = {
                        "threshold": adv_threshold,
                        "threshold_calibration": adv_cal,
                        **adv_metrics,
                        **adv_rank_metrics,
                        **adv_event_metrics,
                        "pred_event_count": len(adv_events),
                        "pred_event_type_counts": event_type_counts(adv_events),
                        "confusion_matrix": adv_cm,
                    }
                    save_json(adv_events, str(report_dir / f"events_{model_name}.json"))

                    adv_rows[f"{model_name}_score"] = adv_scores
                    adv_rows[f"{model_name}_pred"] = adv_pred

                    _save_plot(
                        path=report_dir / f"{model_name}_plot.png",
                        timestamps=ts_adv,
                        signal=signal_adv,
                        labels=y_true_adv,
                        scores=adv_scores,
                        preds=adv_pred,
                        threshold=adv_threshold,
                        title=f"{args.dataset}:{variant} {model_name}",
                    )
                    _save_confusion_plot(
                        path=report_dir / f"{model_name}_confusion_matrix.png",
                        cm=adv_cm,
                        title=f"{args.dataset}:{variant} {model_name} CM",
                    )
    if adv_rows:
        pd.DataFrame(adv_rows).to_csv(report_dir / "advanced_predictions.csv", index=False)

    save_json(metrics_payload, str(report_dir / "metrics.json"))

    _save_plot(
        path=report_dir / "zscore_plot.png",
        timestamps=ts,
        signal=signal,
        labels=y_true,
        scores=z_scores,
        preds=z_pred,
        threshold=z_threshold,
        title=f"{args.dataset}:{variant} Z-score",
    )
    _save_plot(
        path=report_dir / "iforest_plot.png",
        timestamps=ts,
        signal=signal,
        labels=y_true,
        scores=if_scores,
        preds=if_pred,
        threshold=if_threshold,
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
    print(f"[eval] thresholds zscore={z_threshold:.6f} iforest={if_threshold:.6f}")
    print(
        f"[eval] zscore_event_f1={_fmt_metric(z_event_metrics['event_f1'])} "
        f"iforest_event_f1={_fmt_metric(if_event_metrics['event_f1'])}"
    )
    if "lstm_ae" in metrics_payload or "cnn_ae" in metrics_payload:
        lstm_f1 = _fmt_metric(metrics_payload.get("lstm_ae", {}).get("f1"))
        cnn_f1 = _fmt_metric(metrics_payload.get("cnn_ae", {}).get("f1"))
        lstm_thr = _fmt_metric(metrics_payload.get("lstm_ae", {}).get("threshold"))
        cnn_thr = _fmt_metric(metrics_payload.get("cnn_ae", {}).get("threshold"))
        print(f"[eval] lstm_ae_f1={lstm_f1} cnn_ae_f1={cnn_f1}")
        print(f"[eval] thresholds lstm_ae={lstm_thr} cnn_ae={cnn_thr}")


def infer_stream_pi(args) -> None:
    config_path = getattr(args, "config", "configs/base.yaml")
    cfg = _load_runtime_config(dataset=args.dataset, base_config_path=config_path)
    variant = _resolve_variant(args=args, cfg=cfg)
    artifact_dir = Path(args.artifacts_dir) / args.dataset / variant
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    with (artifact_dir / "thresholds.json").open("r", encoding="utf-8") as f:
        thresholds = json.load(f)
    with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    window_size = int(metadata["window_size"])
    ring = RingBuffer(window_size)

    scaler = None
    z_params = None
    iforest = None
    seq_scaler = None
    ae_model = None
    device = str(cfg.get("advanced", {}).get("device", "cpu"))

    if args.model == "zscore":
        scaler = load_pickle(str(artifact_dir / "scaler.pkl"))
        z_params = load_pickle(str(artifact_dir / "zscore_params.pkl"))
    elif args.model == "iforest":
        scaler = load_pickle(str(artifact_dir / "scaler.pkl"))
        iforest = load_pickle(str(artifact_dir / "iforest.pkl"))
    elif args.model in {"lstm_ae", "cnn_ae"}:
        seq_scaler = load_pickle(str(artifact_dir / "seq_scaler.pkl"))
        ckpt = torch.load(artifact_dir / f"{args.model}.pt", map_location=device)
        if args.model == "lstm_ae":
            ae_model = LSTMAutoencoder(
                input_dim=int(ckpt["input_dim"]),
                hidden_dim=int(ckpt.get("hidden_dim", 32)),
            )
        else:
            ae_model = CNNAutoencoder(channels=int(ckpt["channels"]))
        ae_model.load_state_dict(ckpt["state_dict"])
        ae_model = ae_model.to(device).eval()

    if args.source == "local":
        bundle = _load_bundle(args=args, cfg=cfg)
        variant = bundle.variant
        stream_rows = _iter_local_stream_rows(bundle=bundle, cfg=cfg, metadata=metadata)
    else:
        stream_rows = _iter_api_stream_rows(args=args, metadata=metadata)

    threshold = float(thresholds[args.model])
    cal = _resolve_threshold_calibration(cfg=cfg, model_name=args.model)
    effective_threshold = threshold
    calibration_scores: list[float] = []
    threshold_ready = not (cal["enabled"] and cal["mode"] == "warmup_percentile" and cal["warmup_windows"] > 0)
    alerts = 0
    windows_scored = 0
    suppressed_windows = 0

    for timestamp, row in stream_rows:
        ring.push(row)
        if not ring.ready():
            continue

        win = np.asarray(ring.window(), dtype=float)
        score = _score_stream_window(
            model_name=args.model,
            dataset=args.dataset,
            window_arr=win,
            cfg=cfg,
            scaler=scaler,
            z_params=z_params,
            iforest=iforest,
            seq_scaler=seq_scaler,
            ae_model=ae_model,
            device=device,
        )

        windows_scored += 1
        if not threshold_ready:
            calibration_scores.append(score)
            if len(calibration_scores) >= cal["warmup_windows"]:
                effective_threshold = float(np.percentile(calibration_scores, cal["percentile"]))
                threshold_ready = True
            if cal["suppress_during_warmup"]:
                suppressed_windows += 1
                continue

        if score > effective_threshold:
            append_alert(
                log_file=args.log_file,
                timestamp=timestamp,
                score=score,
                threshold=effective_threshold,
                model=args.model,
            )
            alerts += 1

    print(f"[infer] dataset={args.dataset} variant={variant} model={args.model} source={args.source}")
    print(
        f"[infer] base_threshold={threshold:.6f} effective_threshold={effective_threshold:.6f} "
        f"calibration_mode={cal['mode']} suppressed_windows={suppressed_windows}"
    )
    print(f"[infer] windows_scored={windows_scored} alerts={alerts} log_file={args.log_file}")
